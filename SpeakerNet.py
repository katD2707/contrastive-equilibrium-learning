#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import numpy, sys, random
import time, os, importlib
from DatasetLoader import loadWAV
from loss.angleproto import AngleProtoLoss
from loss.protoloss import ProtoLoss
from loss.anglecontrast import AngleContrastiveLoss
from loss.uniform import Uniformity
import numpy as np

class SpeakerNet(nn.Module):

    def __init__(self, lr=0.0001, model="alexnet50", nOut=512, encoder_type='SAP', normalize=True, unif_loss='uniform',
                 sim_loss='anglecontrast', lambda_u=1, lambda_s=1, t=2, sample_type='PoN', **kwargs):
        super(SpeakerNet, self).__init__()

        argsdict = {'nOut': nOut, 'encoder_type': encoder_type}

        SpeakerNetModel = importlib.import_module('models.' + model).__getattribute__(model)
        self.__S__ = SpeakerNetModel(**argsdict).cuda()

        if unif_loss == 'uniform':
            self.__U__ = Uniformity(uniform_t=t, sample_type=sample_type).cuda()
        else:
            raise ValueError('Undefined loss.')

        if sim_loss == 'angleproto':
            self.__L__ = AngleProtoLoss().cuda()
            self.__train_normalize__ = True
            self.__test_normalize__ = True
        elif sim_loss == 'proto':
            self.__L__ = ProtoLoss().cuda()
            self.__train_normalize__ = False
            self.__test_normalize__ = False
        elif sim_loss == 'anglecontrast':
            self.__L__ = AngleContrastiveLoss().cuda()
            self.__train_normalize__ = True
            self.__test_normalize__ = True
        else:
            raise ValueError('Undefined loss.')

        self.lambda_u = lambda_u
        self.lambda_s = lambda_s

        self.__optimizer__ = torch.optim.Adam(
            list(self.__S__.parameters()) + list(self.__U__.parameters()) + list(self.__L__.parameters()), lr=lr)

        self.torchfb = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=0.0, f_max=8000, pad=0, n_mels=40).cuda()
        self.instancenorm = nn.InstanceNorm1d(40).cuda()

        print('Initialised network with nOut %d encoder_type %s, lambda_u = %.2f, lambda_s = %.2f, t = %.2f'
              % (nOut, encoder_type, self.lambda_u, self.lambda_s, t))

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy

        criterion = torch.nn.CrossEntropyLoss()
        conf_labels = torch.LongTensor([1] * stepsize + [0] * stepsize).cuda()

        tstart = time.time()
        print('Start training...')

        for data in loader:                                         #data: [B, n_pairs, n_mels, time]

            self.zero_grad()

            data = data.transpose(0, 1).unsqueeze(2)                # [n_pairs, B, 1, mels, time]

            feat = []
            for inp in data:                                        # [B, 1, mels, time]
                outp = self.__S__.forward(torch.FloatTensor(inp).cuda())    # [B, D]
                if self.__train_normalize__:
                    outp = F.normalize(outp, p=2, dim=1)
                feat.append(outp)
                                                                    # [2, B, D]
            feat = torch.stack(feat, dim=1)                         # [B, 2, D]

            nloss_u, _ = self.__U__.forward(feat, None)
            nloss_s, prec1 = self.__L__.forward(feat, None)

            nloss = self.lambda_u * nloss_u + self.lambda_s * nloss_s

            loss += nloss.detach().cpu()
            top1 += prec1
            counter += 1
            index += stepsize

            nloss.backward()
            self.__optimizer__.step()

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) " % (index))
            sys.stdout.write(
                "Loss %f EER/TAcc %2.3f%% - %.2f Hz" % (loss / counter, top1 / counter, stepsize / telapsed))
            sys.stdout.flush()

        sys.stdout.write("\n")

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, listfilename, print_interval=100, test_path='', num_eval=10, eval_frames=200):

        print(
            'Evaluating with NumEval %d EvalFrames %d Normalize %s' % (num_eval, eval_frames, self.__test_normalize__))

        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):  # or (len(all_scores)==1000)
                    break

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(
                loadWAV(os.path.join(test_path, file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

            with torch.no_grad():

                feat = self.torchfb(inp1) + 1e-6
                feat = self.instancenorm(feat.log()).unsqueeze(1).detach()

                ref_feat = self.__S__.forward(feat).detach().cpu()

            filename = '%06d.wav' % idx

            feats[file] = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d" % (
                idx, len(setfiles), idx / telapsed, ref_feat.size()[1]))

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__test_normalize__:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1),
                                       com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy()
            score = -1 * numpy.mean(dist)

            # dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();
            # score = -1 * numpy.mean(dist);

            # dist = F.cosine_similarity(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();
            # score = numpy.mean(dist);

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz" % (idx, len(lines), idx / telapsed))
                sys.stdout.flush()

        print('\n')

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr'] * alpha
            learning_rate.append(param_group['lr'])

        return learning_rate

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
