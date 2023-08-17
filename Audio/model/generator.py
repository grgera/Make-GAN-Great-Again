
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import *

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1,3,5,7)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])

        for i in range(4):
            conv = weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], 
                                      padding=get_padding(kernel_size, dilation[i])))
            self.convs1.append(conv)
        self.convs1.apply(init_weights)

        for i in range(4):
            conv = weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                          padding=get_padding(kernel_size, 1)))
            self.convs2.append(conv)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.top_k = h.top_k
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_rates = h.upsample_rates
        self.up_kernels = h.upsample_kernel_sizes
        init_ch = h.mel_dim

        self.conv_pre = weight_norm(Conv1d(init_ch, h.upsample_init_channel, 7, 1, padding=3)) 

        self.ups = nn.ModuleList()
        self.nn_up = nn.ModuleList()
        self.resid = nn.ModuleList()

        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_init_channel // (2**i), h.upsample_init_channel // (2**(i+1)),
                                kernel_size=k, stride=u, padding=(k-u)//2)
            ))

            if i >= (self.num_upsamples - self.top_k):
                self.nn_up.append(weight_norm(
                    ConvTranspose1d(init_ch, h.upsample_init_channel // (2**i), kernel_size=self.up_kernels[i-1], 
                                    stride=self.upsample_rates[i-1], 
                                    padding=(self.up_kernels[i-1] - self.upsample_rates[i-1]) // 2)
                ))
                init_ch = h.upsample_init_channel // (2**i)
            
            if i > (self.num_upsamples - self.top_k):
                self.resid.append(nn.Sequential(
                        nn.Upsample(scale_factor=u, mode='nearest'),
                        weight_norm(Conv1d(h.upsample_init_channel // (2**i),
                                           h.upsample_init_channel // (2**(i+1)), 1))
                ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            rb_list = nn.ModuleList()
            ch = h.upsample_init_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                rb_list.append(ResBlock(ch, k, d))
            self.resblocks.append(rb_list)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.nn_up.apply(init_weights)
        self.resid.apply(init_weights)

    def forward(self, x):
        mel = x
        x = self.conv_pre(mel)

        output = None
        for i, resblocks in zip(range(self.num_upsamples), self.resblocks):
            if i >= (self.num_upsamples - self.top_k):
                mel = self.nn_up[i - (self.num_upsamples - self.top_k)](mel)
                x += mel

            if i > (self.num_upsamples - self.top_k):
                if output is not None:
                    output = self.resid[i - (self.num_upsamples - self.top_k) - 1](output)
                else:
                    output = self.resid[i - (self.num_upsamples - self.top_k) - 1](x)

            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for rb in resblocks:
                xs += rb(x)
            x = xs / self.num_kernels
            
            if output is not None:
                output += x

        x = F.leaky_relu(output)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.nn_up:
            remove_weight_norm(l)
        for l in self.resid:
            remove_weight_norm(l[1])
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
