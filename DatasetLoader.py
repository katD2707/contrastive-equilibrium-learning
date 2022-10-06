#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy
import random
import os
import math
import glob
from scipy.io import wavfile
from config import *
from torch.utils.data import Dataset
from torchaudio import transforms
from scipy import signal


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


class wav_split(Dataset):
    def __init__(self,
                 dataset_file_name: str,
                 max_frames,
                 train_path,
                 musan_path,
                 augment_anchor,
                 augment_type):
        self.dataset_file_name = dataset_file_name; #list/youtube-dataset.txt
        self.max_frames = max_frames;

        self.data_dict = {};
        self.data_list = [];
        self.nFiles = 0;

        self.torchfb = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=0.0, f_max=8000, pad=0, n_mels=40);
        self.instancenorm = nn.InstanceNorm1d(40);

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.noiselist = {}                                   # {'music': [file_1, file_2, ...], }
        self.augment_anchor = augment_anchor                  # True
        self.augment_type = augment_type                      # 3
                                                                            # musan_path: MUSAN\musan_split
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))  # MUSAN\musan_split\music\fma\music-fma-0000\*.wav

        for file in augment_files:
            if not file.split('\\')[-4] in self.noiselist:
                self.noiselist[file.split('\\')[-4]] = []
            self.noiselist[file.split('\\')[-4]].append(file)

        self.rir = numpy.load('rir.npy')                                    #[[...],[...],...] (1000, 11200)

        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;
                data = line.split();
                filename = os.path.join(train_path, data[0]);
                self.data_list.append(filename)                             #['train_file_1', ...]

    def __getitem__(self, index):

        audio = loadWAVSplit(self.data_list[index], self.max_frames).astype(numpy.float)    # [2, wav_length]

        augment_profiles = []
        audio_aug = []

        for ii in range(0, 2):

            ## rir profile                          -7                      3
            rir_gains = numpy.random.uniform(SIGPRO_MIN_RANDGAIN, SIGPRO_MAX_RANDGAIN, 1)
            rir_filts = random.choice(self.rir)

            ## additive noise profile
            noisecat = random.choice(self.noisetypes)
            noisefile = random.choice(self.noiselist[noisecat].copy())
            snr = [random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])]

            if self.augment_type == 0 or (ii == 0 and not self.augment_anchor):
                augment_profiles.append({'rir_filt': None, 'rir_gain': None, 'add_noise': None, 'add_snr': None})

            elif self.augment_type == 1:
                augment_profiles.append({'rir_filt': None, 'rir_gain': None, 'add_noise': noisefile, 'add_snr': snr})

            elif self.augment_type == 2:
                ## RIR with 25% chance, otherwise additive noise augmentation
                if random.random() > 0.75:
                    augment_profiles.append(
                        {'rir_filt': rir_filts, 'rir_gain': rir_gains, 'add_noise': None, 'add_snr': None})
                else:
                    augment_profiles.append(
                        {'rir_filt': None, 'rir_gain': None, 'add_noise': noisefile, 'add_snr': snr})

            elif self.augment_type == 3:
                ## RIR and additive noise augmentation
                augment_profiles.append(
                    {'rir_filt': rir_filts, 'rir_gain': rir_gains, 'add_noise': noisefile, 'add_snr': snr})

            else:
                raise ValueError('Invalid augment profile %d' % (self.augment_type))

        audio_aug.append(self.augment_wav(audio[0], augment_profiles[0]))
        audio_aug.append(self.augment_wav(audio[1], augment_profiles[1]))
                                                                    # [2, wav_length]
        audio_aug = numpy.concatenate(audio_aug, axis=0)            # [..., N-mels, wav_length]

        feat = torch.FloatTensor(audio_aug)

        feat = self.torchfb(feat) + 1e-6
        feat = self.instancenorm(feat.log())

        return feat

    def __len__(self):
        return len(self.data_list)

    def augment_wav(self, audio, augment):

        if augment['rir_filt'] is not None:
            audio = gen_echo(audio, augment['rir_filt'], augment['rir_gain'])

        if augment['add_noise'] is not None:

            noiseaudio = loadWAV(augment['add_noise'], self.max_frames, evalmode=False).astype(numpy.float)

            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)

            noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
            audio = audio + noise

        else:

            audio = numpy.expand_dims(audio, 0)

        return audio


def gen_echo(ref, rir, filterGain):
    rir = numpy.multiply(rir, pow(10, 0.1 * filterGain))
    echo = signal.convolve(ref, rir, mode='full')[:len(ref)]

    return echo


def round_down(num, divisor):
    return num - (num % divisor)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio = wavfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = math.floor((max_audio - audiosize + 1) / 2)
        audio = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audio = numpy.append(audio, 0)
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])              # [1, max_audio]

    feat = numpy.stack(feats, axis=0)

    return feat;


def loadWAVSplit(filename, max_frames):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio = wavfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = math.floor((max_audio - audiosize + 1) / 2)
        audio = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audio = numpy.append(audio, 0)
        audiosize = audio.shape[0]

    randsize = audiosize - max_audio
    startframe = random.sample(range(0, randsize+1), 2)
    startframe.sort()
    startframe = numpy.array(startframe)
    assert randsize >= 1

    # for more permutation
    numpy.random.shuffle(startframe)

    feats = []
    for asf in startframe:
        feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0)

    return feat;                                    # [2, max_audio]


def get_data_loader(dataset_file_name, batch_size, max_frames, nDataLoaderThread, train_path, augment_anchor,
                    augment_type, musan_path, **kwargs):
    train_dataset = wav_split(dataset_file_name, max_frames, train_path, musan_path, augment_anchor, augment_type)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nDataLoaderThread,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader
