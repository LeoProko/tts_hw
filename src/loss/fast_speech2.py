import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        mel_output,
        duration_predicted,
        pitch_predicted,
        energy_predicted,
        mel_target,
        length_target,
        pitch_target,
        energy_target,
        *args,
        **kwargs
    ):
        mel_loss = self.mse(mel_output, mel_target)

        duration_predictor_loss = self.mse(
            torch.log1p(duration_predicted.squeeze()),
            torch.log1p(length_target.float()),
        )
        pitch_predictor_loss = self.mse(
            # torch.log1p(pitch_predicted.squeeze()),
            # torch.log1p(pitch_target.float()),
            pitch_predicted.squeeze(),
            pitch_target.float(),
        )
        energy_predictor_loss = self.mse(
            # torch.log1p(energy_predicted.squeeze()),
            # torch.log1p(energy_target.float()),
            energy_predicted.squeeze(),
            energy_target.float(),
        )

        return (
            mel_loss,
            duration_predictor_loss,
            pitch_predictor_loss,
            energy_predictor_loss,
        )
