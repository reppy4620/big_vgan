import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, remove_weight_norm

from .activation import AntiAliasActivation


LRELU_SLOPE = 0.1


class AMPLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size * dilation - dilation)//2, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, dilation=1))

        self.act1 = AntiAliasActivation(channels)
        self.act2 = AntiAliasActivation(channels)

    def forward(self, x):
        y = self.act1(x)
        y = self.conv1(y)
        y = self.act2(y)
        y = self.conv2(y)
        return x + y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([
            AMPLayer(channels, kernel_size, dilation)
            for dilation in dilations
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class BigVGAN(nn.Module):
    def __init__(
        self, 
        in_channel, 
        upsample_initial_channel, 
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3))
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i), 
                        upsample_initial_channel // (2 ** (i + 1)), 
                        kernel_size=k, 
                        stride=u,
                        padding=(k - u) // 2
                    )
                )
            )

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList([
                    AMPBlock(channel, kernel_size=k, dilations=d) 
                    for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                ])
            )
        self.act_post = AntiAliasActivation(channel)
        self.conv_post = weight_norm(nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for up, mrf in zip(self.upsamples, self.mrfs):
            x = up(x)
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.resblocks:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)
