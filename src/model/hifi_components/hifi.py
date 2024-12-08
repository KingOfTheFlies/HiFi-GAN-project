from typing import Any, Dict, List, Callable, Sequence, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class MPDSub(nn.Module):
    def __init__(self, period: int, padding_value: float = 0.0,
                 norm_fn: Callable[[nn.Module], nn.Module] = nn.utils.weight_norm):
        super().__init__()

        self.period = period
        self.padding_value = padding_value

        self.convs = nn.ModuleList([
            nn.Sequential(
                norm_fn(nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU()
            ) for in_channels, out_channels in zip([1, 64, 128, 256, 512], [64, 128, 256, 512, 1024])
        ])

        self.final_conv = norm_fn(nn.Conv2d(1024, 1, kernel_size=(3, 1)))

    def forward(self, input: Tensor, **batch) -> Dict[str, Any]:
        T = input.size(-1)
        new_T = self.period * ((T + self.period - 1) // self.period)
        input = F.pad(input, (0, new_T - T), value=self.padding_value)
        B = input.size(0)
        input = input.view(B, 1, -1, self.period)

        feature_maps = []
        for conv in self.convs:
            input = conv(input)
            feature_maps.append(input)

        output = self.final_conv(input).view(B, -1)
        feature_maps = list(feature_maps)  # Ensure feature_maps is a list

        return {
            'output': output,
            'feature_maps': feature_maps
        }


class MPD(nn.Module):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11), padding_value: float = 0.0):
        super().__init__()
        self.subdiscriminators = nn.ModuleList(
            MPDSub(period=period, padding_value=padding_value) for period in periods
        )

    def forward(self, wave: Tensor, **batch) -> Dict[str, Any]:
        feature_maps, outputs = [], []
        for subdisc in self.subdiscriminators:
            result = subdisc(wave)
            outputs.append(result['output'])
            feature_maps += result['feature_maps']  # Use extend to avoid concatenating with tuples

        return {
            'outputs': outputs,
            'feature_maps': feature_maps
        }


class MSDSub(nn.Module):
    def __init__(self, wave_scale: int = 1, 
                 kernel_sizes: Sequence[int] = (15, 41, 41, 41, 41, 41, 5),
                 strides: Sequence[int] = (1, 2, 2, 4, 4, 1, 1),
                 hidden_channels: Sequence[int] = (128, 128, 256, 512, 1024, 1024, 1024),
                 num_groups: Sequence[int] = (1, 4, 16, 16, 16, 16, 1),
                 norm_fn: Callable[[nn.Module], nn.Module] = nn.utils.weight_norm):
        super().__init__()

        assert len(kernel_sizes) == len(strides) == len(hidden_channels) == len(num_groups)

        self.wave_scale = wave_scale

        self.convs = nn.ModuleList([
            nn.Sequential(
                norm_fn(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                  groups=group, padding=kernel_size // 2)),
                nn.LeakyReLU()
            ) for in_channels, out_channels, kernel_size, stride, group in 
            zip([1] + list(hidden_channels)[:-1], hidden_channels, kernel_sizes, strides, num_groups)
        ])

        self.final_conv = norm_fn(nn.Conv1d(hidden_channels[-1], 1, kernel_size=3, padding=1))

    def forward(self, input: Tensor, **batch) -> Dict[str, Any]:
        if self.wave_scale > 1:
            input = F.avg_pool1d(input, kernel_size=self.wave_scale, stride=self.wave_scale)

        feature_maps = []
        for conv in self.convs:
            input = conv(input)
            feature_maps.append(input)

        output = self.final_conv(input).view(input.size(0), -1)
        feature_maps = list(feature_maps)  # Ensure feature_maps is a list

        return {
            'output': output,
            'feature_maps': feature_maps
        }


class MSD(nn.Module):
    def __init__(self, wave_scales: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.subdiscriminators = nn.ModuleList(
            MSDSub(wave_scale=scale) for scale in wave_scales
        )

    def forward(self, wave: Tensor, **batch) -> Dict[str, Any]:
        feature_maps, outputs = [], []
        for subdisc in self.subdiscriminators:
            result = subdisc(wave)
            outputs.append(result['output'])
            feature_maps += result['feature_maps']  # Use extend to avoid concatenating with tuples

        return {
            'outputs': outputs,
            'feature_maps': feature_maps
        }


class ResSubBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv1d(num_channels, num_channels, kernel_size, dilation=d, padding='same')
            ) for d in dilations
        ])

    def forward(self, input: Tensor, **batch) -> Tensor:
        return self.layers(input) + input


class ResBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, dilations: Sequence[Sequence[int]]):
        super().__init__()
        self.layers = nn.Sequential(*[
            ResSubBlock(num_channels, kernel_size, dilation) for dilation in dilations
        ])

    def forward(self, input: Tensor, **batch) -> Tensor:
        return self.layers(input)


class MRF(nn.Module):
    def __init__(self, num_channels: int, kernel_sizes: Sequence[int], 
                 dilations: Sequence[Sequence[Sequence[int]]]):
        super().__init__()

        assert len(kernel_sizes) == len(dilations)
        self.blocks = nn.ModuleList([
            ResBlock(num_channels, k_size, d) for k_size, d in zip(kernel_sizes, dilations)
        ])

    def forward(self, input: Tensor, **batch) -> Tensor:
        return sum(block(input) for block in self.blocks)


class MRFStack(nn.Module):
    def __init__(self, num_channels: int, transpose_kernel_sizes: Sequence[int], mrf_config: Dict[str, Any]):
        super().__init__()
        layers = []
        out_channels = num_channels

        for kernel in transpose_kernel_sizes:
            in_channels = out_channels
            out_channels //= 2
            layers.append(nn.Sequential(
                nn.LeakyReLU(),
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel, stride=kernel // 2, padding=(kernel - (kernel // 2)) // 2),
                MRF(out_channels, **mrf_config)
            ))

        self.stack = nn.Sequential(*layers)

    def forward(self, input: Tensor, **batch) -> Tensor:
        return self.stack(input)


class HiFiGenerator(nn.Module):
    def __init__(self, mel_channels: int, hidden_channels: int, transpose_kernels: Sequence[int],
                 mrf_config: Dict[str, Any]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(mel_channels, hidden_channels, kernel_size=7, stride=1, padding="same"),
            MRFStack(hidden_channels, transpose_kernels, mrf_config),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels // (2 ** len(transpose_kernels)), 1, kernel_size=7, stride=1, padding="same"),
            nn.Tanh()
        )

    def forward(self, mel_spec: Tensor, **batch) -> Tensor:
        wav = self.model(mel_spec)
        true_length = 22050                     # TODO: исправить
        return wav[..., :true_length]


class HiFiDiscriminator(nn.Module):
    def __init__(self, msd_config: Optional[Dict[str, Any]] = None, mpd_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.msd = MSD(**(msd_config or {}))
        self.mpd = MPD(**(mpd_config or {}))

    def forward(self, wave: Tensor, **batch) -> Dict[str, Any]:
        msd_result = self.msd(wave)
        mpd_result = self.mpd(wave)
        return {
            'msd': {
                'outputs': msd_result['outputs'],
                'feature_maps': msd_result['feature_maps']
            },
            'mpd': {
                'outputs': mpd_result['outputs'],
                'feature_maps': mpd_result['feature_maps']
            }
        }