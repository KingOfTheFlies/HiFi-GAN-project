from .hifi_components.hifi import HiFiDiscriminator, HiFiGenerator

from typing import Any, Dict, List, Optional, Callable, Sequence

import torch
from torch import Tensor

import torch.nn as nn

class HiFiGAN(nn.Module):
    def __init__(self, generator_config, discriminator_config):
        super().__init__()
        self.generator = HiFiGenerator(**generator_config)
        self.discriminator = HiFiDiscriminator(**discriminator_config)

    def forward(self, mel_spec: Tensor, true_wav: Tensor, **batch) -> Dict[str, Any]:
        """
        :param mel_spec: мел-спектрограмма формы (B, freqs, T)
        :return: {
            'wav_generated': сгенерированная волна формы (B, 1, T'),
            'disc_outputs': выходы дискриминатора,
            'disc_feature_maps': карты признаков дискриминатора
        }
        """
        wav_generated = self.generator(mel_spec)
        disc_gen = self.discriminator(wav_generated)
        disc_true = self.discriminator(true_wav)

        return {"model_output": {
                'wav_generated': wav_generated,
                'disc_gen': {
                    'msd': {
                        'outputs': disc_gen['msd']['outputs'],
                        'feature_maps': disc_gen['msd']['feature_maps'],
                    },
                    'mpd': {
                        'outputs': disc_gen['mpd']['outputs'],
                        'feature_maps': disc_gen['mpd']['feature_maps'],
                    }
                },
                'disc_true': {
                    'msd': {
                        'outputs': disc_true['msd']['outputs'],
                        'feature_maps': disc_true['msd']['feature_maps'],
                    },
                    'mpd': {
                        'outputs': disc_true['mpd']['outputs'],
                        'feature_maps': disc_true['mpd']['feature_maps'],
                    }
                }
            }
        }
    
    def generator_parameters(self):
        return self.generator.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info