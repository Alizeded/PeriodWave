import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm
import torchaudio
from einops import rearrange


class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        n_fft, hop_length, win_length = resolution
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=torch.hann_window,
            normalized=True, center=False, pad_mode=None, power=None)

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(2, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), dilation=(2,1), padding=(2, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), dilation=(4,1), padding=(4, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, y):
        fmap = []

        x = self.spec_transform(y)  # [B, 2, Freq, Frames, 2]
        x = torch.cat([x.real, x.imag], dim=1)
        x = rearrange(x, 'b c w t -> b c t w')

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleSTFTDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiScaleSTFTDiscriminator, self).__init__()

        resolutions = [[2048, 512, 2048], [1024, 256, 1024], [512, 128, 512], [256, 64, 256], [128, 32, 128]]

        discs = [DiscriminatorR(resolutions[i], use_spectral_norm=use_spectral_norm) for i in range(len(resolutions))]

        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
