
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn.utils import weight_norm, spectral_norm
from pytorch_wavelets import DWT1D
from utils import *

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.dwt = DWT1D(wave='db1', mode='symmetric')

        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.convs_dwt1 = nn.ModuleList([
            norm_f(Conv1d(2, 1, 1)),
            norm_f(Conv1d(4, 1, 1)),
            norm_f(Conv1d(8, 1, 1))
        ])

        self.convs_dwt2 = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        ])    

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        wavelets =[x]
        x = wave_reshape(x, self.period)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)

            fmap.append(x)

            if i < 3:
                new_wavelets = []

                for wave in wavelets:
                    ca, cd = self.dwt(wave)
                    new_wavelets.append(ca)
                    new_wavelets.append(cd[0])

                wavelets = new_wavelets
                new_wavelets = torch.cat(new_wavelets, dim=1)
                new_wavelets = self.convs_dwt1[i](new_wavelets)
                new_wavelets = wave_reshape(new_wavelets, self.period)
                new_wavelets = self.convs_dwt2[i](new_wavelets)
                x = torch.cat([x, new_wavelets], dim=2)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class ResWiseMPD(nn.Module):
    def __init__(self):
        super(ResWiseMPD, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),                                    
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.dwt = DWT1D(wave='db1', mode='symmetric')

        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 41, 1, padding=20)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        self.convs_dwt1 = nn.ModuleList([
            norm_f(Conv1d(2, 1, 1)),
            norm_f(Conv1d(4, 1, 1)) 
        ])

        self.convs_dwt2 = nn.ModuleList([
            norm_f(Conv1d(1, 128, 1)),
            norm_f(Conv1d(1, 128, 1))
        ])

        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        wavelets = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)

            fmap.append(x)

            if i < 2:
                new_wavelets = []
                for wave in wavelets:
                    ca, cd = self.dwt(wave)
                    new_wavelets.append(ca)
                    new_wavelets.append(cd[0])

                wavelets = new_wavelets
                new_wavelets = torch.cat(new_wavelets, dim=1)
                new_wavelets = self.convs_dwt1[i](new_wavelets)
                new_wavelets = self.convs_dwt2[i](new_wavelets)
                x = torch.cat([x, new_wavelets], dim=2)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class ResWiseMSD(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ResWiseMSD, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),                                  
        ])

        self.dwt = nn.ModuleList([
            DWT1D(wave='db1', mode='symmetric'),
            DWT1D(wave='db1', mode='symmetric')
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        new_y, new_yhat = [y], [y_hat]
        for i, d in enumerate(self.discriminators):
            if i > 0:
                new_wavelets = []
                new_wavelets_hat = []

                for wy, wy_hat in zip(new_y, new_yhat): 
                    ca_y, cd_y = self.dwt[i - 1](wy)
                    ca_yhat, cd_yhat = self.dwt[i - 1](wy_hat)
                    new_wavelets.append(ca_y)
                    new_wavelets.append(cd_y[0])
                    new_wavelets_hat.append(ca_yhat)
                    new_wavelets_hat.append(cd_yhat[0])

                new_y = new_wavelets
                y = torch.cat(new_y, dim=-1)
                new_yhat = new_wavelets_hat
                y_hat = torch.cat(new_yhat, dim=-1)

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def wave_reshape(x, period):
    # 1d to 2d
    b, c, t = x.shape
    if t % period != 0:
        n_pad = period - (t % period)
        x = F.pad(x, (0, n_pad), "reflect")
        t = t + n_pad
    x = x.view(b, c, t // period, period)

    return x