import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        # self.l1_loss = nn.L1Loss()
        self.duration_loss = nn.MSELoss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.duration_loss(
            torch.log1p(duration_predicted),
            torch.log1p(duration_predictor_target.float()),
        )

        return mel_loss, duration_predictor_loss
