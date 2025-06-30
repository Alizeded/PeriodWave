import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d
from torch.nn.utils import weight_norm, remove_weight_norm
from periodwave.commons import get_padding
from periodwave.convnext import ConvNeXtV2Block

LRELU_SLOPE = 0.1


class FinalBlock(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.ResBlocks = torch.nn.Sequential(
            Resblock1D(output_dim, 3, 1),
            Resblock1D(output_dim, 3, 2),
            Resblock1D(output_dim, 3, 4),
        )
        self.final_layer = weight_norm(Conv1d(output_dim, 1, 7, 1, 3, bias=False))

    def forward(self, x):
        x = self.ResBlocks(x)
        x = F.silu(x)
        x = self.final_layer(x)
        return x

    def remove_weight_norm(self):
        for l in self.ResBlocks:
            l.remove_weight_norm()
        remove_weight_norm(self.final_layer)


class Resblock1D(nn.Module):
    def __init__(self, output_dim, kernel_size, dilation):
        super().__init__()

        self.block1 = weight_norm(
            Conv1d(
                output_dim,
                output_dim,
                kernel_size,
                1,
                dilation=dilation,
                padding=get_padding(kernel_size, dilation),
            )
        )
        self.block2 = weight_norm(
            Conv1d(
                output_dim,
                output_dim,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
        )

    def forward(self, x):
        h = F.silu(x)
        h = self.block1(h)
        h = F.silu(h)
        h = self.block2(h)
        x = x + h

        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.block1)
        remove_weight_norm(self.block2)


class Resblock(nn.Module):
    def __init__(self, output_dim, kernel_size, dilation):
        super().__init__()

        self.block1 = weight_norm(
            Conv2d(
                output_dim,
                output_dim,
                (kernel_size, 1),
                (1, 1),
                dilation=(dilation, 1),
                padding=(get_padding(kernel_size, dilation), 0),
            )
        )
        self.block2 = weight_norm(
            Conv2d(
                output_dim,
                output_dim,
                (kernel_size, 1),
                (1, 1),
                dilation=(1, 1),
                padding=(get_padding(kernel_size, 1), 0),
            )
        )

    def forward(self, x):
        h = F.silu(x)
        h = self.block1(h)
        h = F.silu(h)
        h = self.block2(h)
        x = x + h

        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.block1)
        remove_weight_norm(self.block2)


class DownConv2D(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, stride, hidden_dim, act=False
    ):
        super().__init__()
        if act:
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()

        self.down = weight_norm(
            Conv2d(
                input_dim,
                output_dim,
                (kernel_size, 1),
                (stride, 1),
                padding=(get_padding(kernel_size, 1), 0),
                bias=False,
            )
        )

        self.mlp = torch.nn.Sequential(
            nn.SiLU(), torch.nn.Linear(hidden_dim, output_dim)
        )

        self.ResBlocks = torch.nn.Sequential(
            Resblock(output_dim, 3, 1), Resblock(output_dim, 3, 2)
        )

    def forward(self, x, time_emb):
        x = self.act(x)
        x = self.down(x)
        x += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.ResBlocks(x)

        return x

    def remove_weight_norm(self):
        for l in self.ResBlocks:
            l.remove_weight_norm()

        remove_weight_norm(self.down)


class MelCondConv2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()

        self.down = weight_norm(
            Conv2d(
                input_dim,
                output_dim,
                (kernel_size, 1),
                (stride, 1),
                padding=(get_padding(kernel_size, 1), 0),
                bias=False,
            )
        )
        self.ResBlocks = torch.nn.Sequential(
            Resblock(output_dim, 3, 1),
            Resblock(output_dim, 3, 2),
            Resblock(output_dim, 3, 4),
        )

    def forward(self, x, mel_cond):
        x = F.silu(x)
        x = self.down(x)
        x += mel_cond[:, :, : x.shape[2], :]
        x = self.ResBlocks(x)

        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.down)
        for l in self.ResBlocks:
            l.remove_weight_norm()


class UpConv2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, hidden_dim):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            nn.SiLU(), torch.nn.Linear(hidden_dim, output_dim)
        )
        self.ResBlocks = torch.nn.Sequential(
            Resblock(output_dim, 3, 1), Resblock(output_dim, 3, 2)
        )
        self.up = weight_norm(
            ConvTranspose2d(
                input_dim,
                output_dim,
                (stride * 2, 1),
                (stride, 1),
                padding=(stride // 2, 0),
                bias=False,
            )
        )

    def forward(self, x, skip, time_emb, s_w=1, b_w=1):
        x = F.silu(x)
        x = self.up(x)
        x = b_w * x + s_w * skip + self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.ResBlocks(x)

        return x

    def remove_weight_norm(self):
        for l in self.ResBlocks:
            l.remove_weight_norm()

        remove_weight_norm(self.up)


class GeneratorP(torch.nn.Module):
    def __init__(
        self,
        period,
        kernel_size=9,
        stride=4,
        use_spectral_norm=False,
        final_dim=32,
        hidden_dim=512,
    ):
        super(GeneratorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        self.final_dim = final_dim

        self.downs = nn.ModuleList([
            DownConv2D(1, hidden_dim // 16, 1, 1, hidden_dim, act=False),
            DownConv2D(
                hidden_dim // 16,
                hidden_dim // 8,
                kernel_size,
                stride,
                hidden_dim,
                act=True,
            ),
            DownConv2D(
                hidden_dim // 8,
                hidden_dim // 4,
                kernel_size,
                stride,
                hidden_dim,
                act=True,
            ),
        ])
        self.mids = nn.ModuleList([
            MelCondConv2D(hidden_dim // 4, hidden_dim, kernel_size, stride)
        ])
        self.ups = nn.ModuleList([
            UpConv2D(hidden_dim, hidden_dim // 4, kernel_size, stride, hidden_dim),
            UpConv2D(hidden_dim // 4, hidden_dim // 8, kernel_size, stride, hidden_dim),
            UpConv2D(hidden_dim // 8, final_dim, kernel_size, stride, hidden_dim),
        ])

    def forward(self, x, time_emb, mel_cond, i, s_w=1, b_w=1):
        """
        x = concat[xt, 1, t]
        xt = [b,1,t]
        mel_con = [b,1,t] shaped from [b,256,t//256]
        t = [b,1,t] expanded from [b,1]
        """
        # 1d to 2d
        b, c, t = x.shape
        ori_t = t
        if t % (64 * self.period[i]) != 0:  # pad first
            n_pad = self.period[i] * 64 - (t % (self.period[i] * 64))
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period[i], self.period[i])

        skips = []
        for l in self.downs:
            x = l(x, time_emb)
            skips.append(x)

        for l in self.mids:
            x = l(x, mel_cond)

        for l in self.ups:
            x = l(x, skips.pop(), time_emb, s_w=s_w, b_w=b_w)

        x = x.reshape(b, self.final_dim, -1)

        return x[:, :, :ori_t]

    def remove_weight_norm(self):
        for l in self.downs:
            l.remove_weight_norm()
        for l in self.mids:
            l.remove_weight_norm()
        for l in self.ups:
            l.remove_weight_norm()


class MultiPeriodGenerator(torch.nn.Module):
    def __init__(
        self,
        use_spectral_norm=False,
        periods=[1, 2, 3, 5, 7],
        final_dim=32,
        hidden_dim=512,
    ):
        super(MultiPeriodGenerator, self).__init__()

        periods = periods
        self.periods_len = len(periods)
        self.mpg = GeneratorP(
            periods,
            use_spectral_norm=use_spectral_norm,
            final_dim=final_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x, time_emb, mel_cond, s_w=1, b_w=1):
        x_p_s = []
        for i in range(self.periods_len):
            x_p = self.mpg(x, time_emb[:, i, :], mel_cond[i], i, s_w=s_w, b_w=b_w)
            x_p_s.append(x_p)

        return x_p_s

    def remove_weight_norm(self):
        self.mpg.remove_weight_norm()


class MelSpectrogramUpsampler(nn.Module):
    def __init__(self, n_mel, periods=[1, 2, 3, 5, 7], hidden_dim=512):
        super().__init__()

        self.embed = torch.nn.Conv1d(128, hidden_dim, 7, stride=1, padding=3)

        self.convnext_block = torch.nn.ModuleList()
        self.block_num = 8
        for i in range(self.block_num):
            self.convnext_block.append(
                ConvNeXtV2Block(hidden_dim, int(hidden_dim * 3), drop_path=0.1)
            )

        self.up = weight_norm(
            ConvTranspose1d(hidden_dim, hidden_dim // 2, 11, 5, padding=3)
        )

        self.cond_num = 4
        self.cond = torch.nn.ModuleList()
        for i in range(self.cond_num):
            self.cond.append(
                ConvNeXtV2Block(hidden_dim // 2, int(hidden_dim * 2), drop_path=0.1)
            )

        self.cond_block = nn.ModuleList()
        self.periods = periods
        self.len_periods = int(len(periods))
        for p in periods:
            self.cond_block.append(
                weight_norm(
                    Conv2d(
                        hidden_dim // 2,
                        hidden_dim,
                        (p * 2, 1),
                        (p, 1),
                        padding=(get_padding(p * 2, 1), 0),
                    )
                )
            )

    def forward(self, x):
        x = self.embed(x)  # [B,512, T//1920] 12.5Hz
        residual = x
        for i in range(self.block_num):
            x = self.convnext_block[i](x)

        x = x + residual

        x = F.silu(x)
        x = self.up(x)  # [B,512, T//320] 75Hz

        for i in range(self.cond_num):
            x = self.cond[i](x)

        x = F.silu(x)
        x = x.unsqueeze(-1)

        mel_conds = []
        for i in range(self.len_periods):
            n_pad = self.periods[i] - ((x.shape[-1]) % (self.periods[i]))
            mel_cond = F.pad(x, (0, 0, 0, n_pad, 0, 0), "reflect")
            mel_conds.append(self.cond_block[i](mel_cond))

        return mel_conds

    def remove_weight_norm(self):
        remove_weight_norm(self.up)
        for l in self.cond_block:
            remove_weight_norm(l)
