import torch
import torch.nn as nn
from typing import List, Tuple,Optional
import math


class MWT(nn.Module):
    def __init__(self, dim: int, level: int):
        super().__init__()
        low_filter = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1 / torch.sqrt(torch.tensor(2.0))]).view(
            (1, 1, -1, 1))
        self.low_filter = low_filter.repeat(dim, 1, 1, 1).to('cuda')
        high_filter = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), -1 / torch.sqrt(torch.tensor(2.0))]).view(
            (1, 1, -1, 1))
        self.high_filter = high_filter.repeat(dim, 1, 1, 1).to('cuda')

        self._level = level

    def forward(self, x: torch.Tensor):
        b, d, t, m = x.shape
        low = self.low_filter
        high = self.high_filter
        res = []
        cur = x
        for _ in range(self._level):
            res_low = nn.functional.conv2d(cur, low, stride=(low.shape[2], 1), groups=d)
            res_high = nn.functional.conv2d(cur, high, stride=(high.shape[2], 1), groups=d)
            res.append(res_low)
            res.append(res_high)
            cur = res_low
        return res


class Modulation(nn.Module):
    def __init__(self, dim: int, num_modality: int, level: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim // 2)
        self.projs = nn.ModuleList()
        for _ in range(level * 2):
            self.projs.append(
                nn.Conv2d(num_modality, num_modality, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            )

        self._level = level
        self.MWT = MWT(dim // 2, level)
        self._init_weights()

    def forward(self, x: torch.Tensor):
        b, c, t, m = x.shape
        u, v = x.chunk(2, dim=1)
        v = self.norm(v)
        v_wavelets = self.MWT(v)
        for i in range(self._level * 2):
            v_wavelets[i] = torch.transpose(v_wavelets[i], 1, 3)
            v_wavelets[i] = self.projs[i](v_wavelets[i])
            v_wavelets[i] = torch.transpose(v_wavelets[i], 1, 3)
            v_wavelets[i] = v_wavelets[i].unsqueeze(2)
            v_wavelets[i] = v_wavelets[i].expand(-1, -1, 2 ** (i // 2 + 1), -1, -1).reshape(b, c // 2, t, m)

        res_v = sum(v_wavelets) / len(v_wavelets)
        u = u * res_v
        return u

    def _init_weights(self):
        for p in self.projs:
            nn.init.constant_(p.weight, 0)
            nn.init.constant_(p.bias, 1)


class PhysiologicalBlock(nn.Module):
    def __init__(self, dim, dim_expansion, prob, num_modality, level):
        super().__init__()

        self.norm = nn.GroupNorm(1, dim)
        self.proj1 = nn.Conv2d(dim, dim_expansion, kernel_size=1, stride=1, padding=0)
        self.activation = nn.GELU()
        self.mod = Modulation(dim_expansion, num_modality, level)
        self.proj2 = nn.Conv2d(dim_expansion // 2, dim, kernel_size=1, stride=1, padding=0)
        self.prob = prob
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training and torch.equal(self.sampler.sample(), torch.zeros(1)):
            return x
        residual = x.clone()
        x = self.norm(x)
        x = self.proj1(x)
        x = self.activation(x)
        x = self.mod(x)
        x = self.proj2(x)
        return x + residual


class PhysiologicalInteraction(nn.Module):
    def __init__(
            self,
            dim: int, dim_expansion: Optional[int]=None,
            num_modality: int=8, level: int=5,
            num_blocks: int=4,
            prob_range=(1, 0.5)
    ):
        super().__init__()
        if dim_expansion is None:
            dim_expansion=dim*4
        self.probs = torch.linspace(prob_range[0], prob_range[1], num_blocks)

        self.blocks = nn.ModuleList()

        for prob in self.probs:
            self.blocks.append(PhysiologicalBlock(dim, dim_expansion, prob, num_modality**2, level))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SubmodalityEmbed(nn.Module):
    def __init__(self, in_ch, dim: int = 64, in_size: int = 128, in_size_t: int = 288, num_modality: int = 8, ratio: float = 0.04):
        super().__init__()
        channels = [min(dim, 16 * (2 ** i)) for i in range(int(math.log2(in_size // num_modality)) + 1)]
        m = 3
        M = 7
        kernel_s = max(m, min(M, int(in_size * ratio)))
        kernel_t = max(m, min(M, int(in_size_t * ratio)))
        kernel_s = kernel_s if kernel_s % 2 == 1 else kernel_s + 1
        kernel_t = kernel_t if kernel_t % 2 == 1 else kernel_t + 1
        stem_kernel = (1, kernel_s, kernel_s)
        sub_kernel = (kernel_t, kernel_s, kernel_s)

        self.layers = nn.Sequential(nn.Conv3d(in_channels=in_ch, out_channels=channels[0], kernel_size=stem_kernel, stride=(1, 1, 1),padding=self._calc_padding(stem_kernel)), nn.GroupNorm(1, channels[0]), nn.ELU())
        for i in range(1, len(channels)):
            self.layers.append(nn.Sequential(nn.Conv3d(channels[i - 1], channels[i], kernel_size=sub_kernel, stride=(1, 2, 2),
                              padding=self._calc_padding(sub_kernel)), nn.GroupNorm(1, channels[i]), nn.ELU()))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds
        x = self.layers(x)
        b, d, t, h, w = x.shape
        return x.view(b, d, t, h * w)

    def _calc_padding(self, k: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple((s - 1) // 2 for s in k)


class RFEncoder(nn.Module):
    def __init__(self, in_ch=10):
        super().__init__()
        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, 32, 7, stride=1, padding=3),
            nn.GroupNorm(1, 32),
            nn.ELU(),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.GroupNorm(1, 64),
            nn.ELU(),
        )
        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 7, stride=1, padding=3),
            nn.GroupNorm(1, 128),
            nn.ELU(),
        )
        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 7, stride=1, padding=3),
            nn.GroupNorm(1, 256),
            nn.ELU(),
        )
        self.ConvBlock5_mean = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, 7, stride=1, padding=3),
        )
        self.downsample1 = torch.nn.MaxPool1d(kernel_size=2)
        self.downsample2 = torch.nn.MaxPool1d(kernel_size=2)

        self.TConvBlock1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(512, 256, 7, stride=1, padding=3),
            nn.GroupNorm(1, 256),
            nn.ELU(),
        )

        self.TConvBlock2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(256, 128, 7, stride=1, padding=3),
            nn.GroupNorm(1, 128),
            nn.ELU(),
        )
        self.TConvBlock3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(128, 64, 7, stride=1, padding=3),
        )

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.downsample1(x)
        x = self.ConvBlock4(x)
        x = self.downsample2(x)
        x = self.ConvBlock5_mean(x)

        x = self.TConvBlock1(x)
        x = self.TConvBlock2(x)
        x = self.TConvBlock3(x)
        b,d,t=x.shape
        return x.view(b,d,t,1)

class Predictor(nn.Module):
    def __init__(self,dim:int=64):
        super().__init__()
        self.predict=nn.Sequential(
            nn.GroupNorm(1,dim),
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self,x):
        x = self.predict(x)
        x=torch.squeeze(x,dim=1)
        x = torch.mean(x, dim=2)
        return x




