import torch
import torch.nn.functional as F
from torch import nn
from src.utils.mel_utils import MelSpectrogram, MelSpectrogramConfig


class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_fm=2.0, lambda_mel=45.0):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.mel_loss = nn.L1Loss()  # L1 Loss для мел-спектрограмм

        self.mel_spec_conf = MelSpectrogramConfig()
        self.mel_transform = MelSpectrogram(self.mel_spec_conf)

    def adversarial_loss(self, disc_outputs, target_is_real):
        losses = []
        target_value = 1.0 if target_is_real else 0.0
        for output in disc_outputs:
            targets = torch.full_like(output, target_value, device=output.device)
            losses.append(F.mse_loss(output, targets))
        return sum(losses) / len(losses)

    def feature_matching_loss(self, gen_feature_maps, true_feature_maps):
        losses = []
        for gen, true in zip(gen_feature_maps, true_feature_maps):
            min_len = min(gen.shape[-1], true.shape[-1])
            losses.append(F.l1_loss(gen[..., :min_len], true[..., :min_len]))
        return sum(losses) / len(losses)

    def mel_spectrogram_loss(self, gen_wav, true_wav):
        gen_mel = self.mel_transform(gen_wav)
        true_mel = self.mel_transform(true_wav)
        return self.mel_loss(gen_mel, true_mel)

    def forward(self, model_output, true_wav, **batch):
        # Adversarial Loss для генератора
        gen_adv_loss_msd = self.adversarial_loss(model_output['disc_gen']['msd']['outputs'], target_is_real=True)
        gen_adv_loss_mpd = self.adversarial_loss(model_output['disc_gen']['mpd']['outputs'], target_is_real=True)
        gen_adv_loss = gen_adv_loss_msd + gen_adv_loss_mpd

        # Adversarial Loss для дискриминатора
        disc_adv_loss_real_msd = self.adversarial_loss(model_output['disc_true']['msd']['outputs'], target_is_real=True)
        disc_adv_loss_real_mpd = self.adversarial_loss(model_output['disc_true']['mpd']['outputs'], target_is_real=True)
        disc_adv_loss_fake_msd = self.adversarial_loss(model_output['disc_gen']['msd']['outputs'], target_is_real=False)
        disc_adv_loss_fake_mpd = self.adversarial_loss(model_output['disc_gen']['mpd']['outputs'], target_is_real=False)
        disc_adv_loss = disc_adv_loss_real_msd + disc_adv_loss_real_mpd + disc_adv_loss_fake_msd + disc_adv_loss_fake_mpd

        # Feature Matching Loss
        fm_loss = self.feature_matching_loss(
            model_output['disc_gen']['msd']['feature_maps'] + model_output['disc_gen']['mpd']['feature_maps'],
            model_output['disc_true']['msd']['feature_maps'] + model_output['disc_true']['mpd']['feature_maps']
        )

        # Mel-Spectrogram Loss
        mel_loss = self.mel_spectrogram_loss(model_output['wav_generated'], true_wav)

        # Weighted sum
        gen_loss = gen_adv_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_loss

        loss = gen_loss + disc_adv_loss
        return {
            'loss': loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_adv_loss,
            'gen_adv_loss': gen_adv_loss,
            'fm_loss': fm_loss,
            'mel_loss': mel_loss,
        }
