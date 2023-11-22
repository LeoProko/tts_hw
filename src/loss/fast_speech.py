import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        # self.l1_loss = nn.L1Loss()
        self.duration_loss = nn.MSELoss()

    def forward(
        self, mel_output, duration_predicted, mel_target, length_target, *args, **kwargs
    ):
        mel_loss = self.mse_loss(mel_output, mel_target)

        print(duration_predicted.shape, length_target.shape)
        duration_predictor_loss = self.duration_loss(
            torch.log1p(duration_predicted),
            torch.log1p(length_target.float()),
        )

        return mel_loss, duration_predictor_loss
