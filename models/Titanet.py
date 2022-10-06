import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv1dSamePadding(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, x):
        padding = (
                          self.stride[0] * (x.shape[-1] - 1)
                          - x.shape[-1]
                          + self.kernel_size[0]
                          + (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
                  ) // 2
        return self._conv_forward(
            F.pad(x, (padding, padding)),
            self.weight,
            self.bias,
        )


class DepthwiseConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True,
                 device=None,
                 dtype=None,
                 ):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSamePadding(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            Conv1dSamePadding(
                in_channels,
                out_channels,
                kernel_size=1,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 activation='relu',
                 dropout=0.,
                 depthwise=False, ):
        super(ConvBlock1d, self).__init__()
        assert activation is None or activation in (
            "relu",
            "tanh",

        ), "Incompatible activation function"

        conv_module = DepthwiseConv1d if depthwise else Conv1dSamePadding
        modules = [
            conv_module(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation is not None:
            modules += [nn.ReLU() if activation == "relu" else nn.Tanh()]
        if dropout > 0:
            modules += [nn.Dropout(p=dropout)]
        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(output_size=1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        squeezed = self.squeeze(x).squeeze(-1)

        excited = self.excitation(squeezed).unsqueeze(-1)

        return x * excited.expand_as(x)


class Squeeze(nn.Module):
    def __init(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Titanet(nn.Module):
    TARGET_PARAMS = {"s": 6.4, "m": 13.4, "l": 25.3}

    def __init__(self,
                 n_mels,
                 n_mega_blocks,
                 n_sub_blocks,
                 encoder_hidden_size,
                 encoder_output_size,
                 embedding_size,
                 mega_block_kernel_size,
                 prolog_kernel_size=3,
                 epilog_kernel_size=1,
                 attention_hidden_size=128,
                 se_reduction=16,
                 simple_pool=False,
                 loss_function=None,
                 dropout=0.5,
                 device="cpu",
                 ):
        super(Titanet, self).__init__()

        self.encoder = Encoder(
            n_mels,
            n_mega_blocks,
            n_sub_blocks,
            encoder_hidden_size,
            encoder_output_size,
            mega_block_kernel_size,
            prolog_kernel_size=prolog_kernel_size,
            epilog_kernel_size=epilog_kernel_size,
            se_reduction=se_reduction,
            dropout=dropout,
        )

        self.decoder = Decoder(
            encoder_output_size,
            attention_hidden_size,
            embedding_size,
            simple_pool=simple_pool,
        )

        self.conv1d = nn.Conv2d()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 n_mels,
                 n_mega_blocks,
                 n_sub_blocks,
                 hidden_size,
                 output_size,
                 mega_block_kernel_size,
                 prolog_kernel_size=3,
                 epilog_kernel_size=1,
                 se_reduction=16,
                 dropout=0.5,
                 ):
        super(Encoder, self).__init__()

        self.prolog = ConvBlock1d(n_mels, hidden_size, prolog_kernel_size)
        self.mega_blocks = nn.Sequential(*[
            MegaBlock(hidden_size,
                      hidden_size,
                      mega_block_kernel_size,
                      n_sub_blocks,
                      se_reduction=se_reduction,
                      dropout=dropout, )
            for _ in range(n_mega_blocks)
        ])
        self.epilog = ConvBlock1d(hidden_size, output_size, epilog_kernel_size)

    def forward(self, x):
        x = self.prolog(x)
        x = self.mega_blocks(x)
        x = self.epilog(x)
        return x


class MegaBlock(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size,
                 n_sub_blocks,
                 se_reduction=16,
                 dropout=0.5,
                 ):
        super(MegaBlock, self).__init__()

        self.dropout = dropout

        channels = [input_size] + [output_size] * n_sub_blocks
        self.sub_blocks = nn.Sequential(*[
            ConvBlock1d(
                in_channels,
                out_channels,
                kernel_size,
                activation="relu",
                dropout=dropout,
                depthwise=True,
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ],
            SqueezeExcitation(output_size, reduction=se_reduction)
        )

        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, x):
        y = self.sub_blocks(x)
        z = self.skip_connection(x)

        return F.dropout(F.relu(y + z), p=self.dropout, training=self.training)


class Decoder(nn.Module):
    def __init__(self,
                 encoder_output_size,
                 attention_hidden_size,
                 embedding_size,
                 simple_pool=False,
                 ):
        super(Decoder, self).__init__()

        if simple_pool:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                Squeeze(-1),
                nn.Linear(encoder_output_size, encoder_output_size * 2),
            )
        else:
            self.pool = nn.Sequential(
                AttentiveStatsPooling(encoder_output_size, attention_hidden_size),
                nn.BatchNorm1d(encoder_output_size * 2),
            )

        self.linear = nn.Sequential(
            nn.Linear(encoder_output_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.linear(x)

        # Out: [B, E]
        return x


class AttentiveStatsPooling(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-6):
        super(AttentiveStatsPooling, self).__init__()

        self.eps = eps

        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # input: [B, DE, T]
        energies = self.out_linear(nn.Tanh(self.in_linear(x.transpose(1, 2)))).transpose(1, 2)

        alphas = torch.softmax(energies, dim=2)

        means = torch.sum(alphas * x, dim=2)

        residuals = torch.sum(alphas * x ** 2, dim=2) - means ** 2
        stds = torch.sqrt(residuals.clamp(min=self.eps))

        return torch.cat([means, stds], dim=1)
