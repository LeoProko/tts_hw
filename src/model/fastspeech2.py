import torch
from torch import nn
import torch.nn.functional as F

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class FFTBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        num_heads,
        dropout,
        fft_kernel_size_1,
        fft_padding_1,
        fft_kernel_size_2,
        fft_padding_2,
    ):
        super().__init__()

        self.mha = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.conv = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(
                emb_dim,
                hidden_dim,
                kernel_size=fft_kernel_size_1,
                padding=fft_padding_1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                emb_dim,
                kernel_size=fft_kernel_size_2,
                padding=fft_padding_2,
            ),
            Transpose(1, 2),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x, attn_mask):
        x = x + self.norm(self.mha(x, x, x, attn_mask=attn_mask)[0])
        x = x + self.conv(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        vocab_size,
        max_seq_len,
        hidden_dim,
        num_heads,
        n_layers,
        dropout,
        fft_kernel_size_1,
        fft_padding_1,
        fft_kernel_size_2,
        fft_padding_2,
    ):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.positions = nn.Embedding(max_seq_len + 1, emb_dim, padding_idx=PAD_IDX)
        self.num_heads = num_heads

        self.layers = nn.Sequential(
            *[
                FFTBlock(
                    emb_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    fft_kernel_size_1,
                    fft_padding_1,
                    fft_kernel_size_2,
                    fft_padding_2,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, tokens, pos):
        x = self.embeddings(tokens) + self.positions(pos)

        attn_mask = tokens.eq(PAD_IDX).unsqueeze(1).expand(-1, tokens.size(1), -1)
        attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        for i in range(len(self.layers)):
            x = self.layers[i](x, attn_mask)

        return x


class PredictorBlock(nn.Module):
    def __init__(self, emb_dim, hidden_dim, kernel_size, dropout):
        super().__init__()

        self.net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(emb_dim, hidden_dim, kernel_size=kernel_size, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Transpose(-1, -2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, x, alpha):
        x = self.net(x) * alpha

        # if not self.training:
        #     x = x.unsqueeze(0)

        return x


class LengthRegulator(nn.Module):
    def __init__(self, emb_dim, hidden_dim, kernel_size, dropout):
        super().__init__()

    @staticmethod
    def create_alignment(base_mat, duration_predictor_output):
        N, L = duration_predictor_output.shape
        for i in range(N):
            count = 0
            for j in range(L):
                for k in range(duration_predictor_output[i][j]):
                    base_mat[i][count + k][j] = 1
                count = count + duration_predictor_output[i][j]
        return base_mat

    def LR(self, x, duration_predictor_output, mel_max_len=None):
        expand_max_len = max(
            torch.max(torch.sum(duration_predictor_output, -1), -1)[0].item(), 1
        )
        alignment = torch.zeros(
            duration_predictor_output.size(0),
            expand_max_len,
            duration_predictor_output.size(1),
        ).numpy()
        alignment = self.create_alignment(
            alignment, duration_predictor_output.cpu().numpy()
        )
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_len:
            output = F.pad(output, (0, 0, 0, mel_max_len - output.size(1), 0, 0))

        return output

    def forward(self, x, alpha, duration_target, duration, mel_max_len=None):
        if duration_target is not None:
            output = self.LR(x, duration_target, mel_max_len)
            return output

        duration = (duration * alpha + 0.5).int()

        output = self.LR(x, duration.squeeze(-1))

        mel_pos = torch.stack(
            [torch.tensor([i + 1 for i in range(output.size(1))])],
        ).long()

        return output, mel_pos


class VarianceApapter(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        kernel_size,
        dropout,
    ):
        super().__init__()

        self.duration_predictor = PredictorBlock(
            emb_dim, hidden_dim, kernel_size, dropout
        )
        self.length_regulator = LengthRegulator(
            emb_dim, hidden_dim, kernel_size, dropout
        )
        self.pitch_predictor = PredictorBlock(emb_dim, hidden_dim, kernel_size, dropout)
        self.energy_predictor = PredictorBlock(
            emb_dim, hidden_dim, kernel_size, dropout
        )

        self.pitch_emb = nn.Embedding(256, emb_dim)
        self.energy_emb = nn.Embedding(256, emb_dim)

        # precalculated statistics
        self.pitch_bins = nn.Parameter(
            torch.linspace(70.0, 800.0, 255),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(0.0, 1050.0, 255),
            requires_grad=False,
        )

    def forward(
        self,
        x,
        duration_alpha,
        pitch_alpha,
        energy_alpha,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        length_target=None,
    ):
        duration_predictor_output = self.duration_predictor(x, duration_alpha)

        if self.training:
            pitch_predictor_output = self.pitch_predictor(x, pitch_alpha)
            pitch_emb = self.pitch_emb(
                torch.bucketize(pitch_target.detach(), self.pitch_bins.detach())
            )

            energy_predictor_output = self.energy_predictor(x, energy_alpha)
            energy_emb = self.energy_emb(
                torch.bucketize(energy_target.detach(), self.energy_bins.detach())
            )

            print(
                x.shape,
                energy_target.shape,
                energy_predictor_output.shape,
                energy_emb.shape,
            )

            x = x + pitch_emb + energy_emb

            mel_output = self.length_regulator(
                x, duration_alpha, length_target, None, max_len
            )

            return (
                mel_output + pitch_emb + energy_emb,
                duration_predictor_output,
                pitch_predictor_output,
                energy_predictor_output,
            )

        mel_output, mel_pos = self.length_regulator(
            x, duration_alpha, None, duration_predictor_output, None
        )
        pitch_predictor_output = self.pitch_predictor(mel_output, pitch_alpha)
        pitch_emb = self.pitch_emb(
            torch.bucketize(pitch_predictor_output.detach(), self.pitch_bins.detach())
        )

        energy_predictor_output = self.energy_predictor(mel_output, energy_alpha)
        energy_emb = self.energy_emb(
            torch.bucketize(energy_predictor_output.detach(), self.energy_bins.detach())
        )

        return (
            mel_output + pitch_emb.squeeze(2) + energy_emb.squeeze(2),
            mel_pos,
            duration_predictor_output,
            pitch_predictor_output,
            energy_predictor_output,
        )


class Decoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        max_seq_len,
        hidden_dim,
        num_heads,
        n_layers,
        dropout,
        fft_kernel_size_1,
        fft_padding_1,
        fft_kernel_size_2,
        fft_padding_2,
    ):
        super().__init__()

        self.num_heads = num_heads

        self.positions = nn.Embedding(max_seq_len + 1, emb_dim, padding_idx=PAD_IDX)

        self.layers = nn.Sequential(
            *[
                FFTBlock(
                    emb_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    fft_kernel_size_1,
                    fft_padding_1,
                    fft_kernel_size_2,
                    fft_padding_2,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_out, enc_pos):
        attn_mask = enc_pos.eq(PAD_IDX).unsqueeze(1).expand(-1, enc_pos.size(1), -1)
        attn_mask = attn_mask.repeat(self.num_heads, 1, 1).to(enc_out.device)
        # non_pad_mask = enc_pos.ne(PAD_IDX).type(torch.float).unsqueeze(-1)

        x = enc_out + self.positions(enc_pos.to(enc_out.device))

        for i in range(len(self.layers)):
            x = self.layers[i](x, attn_mask)

        return x


class FastSpeech2(nn.Module):
    def __init__(
        self,
        emb_dim,
        vocab_size,
        max_seq_len,
        hidden_dim,
        num_heads,
        n_layers,
        num_mels,
        dropout,
        fft_kernel_size_1,
        fft_padding_1,
        fft_kernel_size_2,
        fft_padding_2,
        predictor_kernel_size,
    ):
        super().__init__()

        self.encoder = Encoder(
            emb_dim,
            vocab_size,
            max_seq_len,
            hidden_dim,
            num_heads,
            n_layers,
            dropout,
            fft_kernel_size_1,
            fft_padding_1,
            fft_kernel_size_2,
            fft_padding_2,
        )
        self.variance_adapter = VarianceApapter(
            emb_dim,
            hidden_dim,
            predictor_kernel_size,
            dropout,
        )
        self.decoder = Decoder(
            emb_dim,
            max_seq_len,
            hidden_dim,
            num_heads,
            n_layers,
            dropout,
            fft_kernel_size_1,
            fft_padding_1,
            fft_kernel_size_2,
            fft_padding_2,
        )
        self.mel_linear = nn.Linear(emb_dim, num_mels)

    @staticmethod
    def get_mask_from_lengths(lengths, max_len=None):
        if max_len == None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len, 1, device=lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()

        return mask

    def mask_tensor(self, mel_output, position, mel_max_len):
        lengths = torch.max(position, -1)[0]
        mask = ~self.get_mask_from_lengths(lengths, max_len=mel_max_len)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.0)

    def forward(
        self,
        src_seq,
        src_pos,
        duration_alpha=1.0,
        pitch_alpha=1.0,
        energy_alpha=1.0,
        mel_pos=None,
        mel_max_len=None,
        pitch_target=None,
        energy_target=None,
        length_target=None,
        *args,
        **kwargs
    ):
        x = self.encoder(src_seq, src_pos)

        if self.training:
            (
                mel_output,
                duration_predictor_output,
                pitch_predictor_output,
                energy_predictor_output,
            ) = self.variance_adapter(
                x,
                duration_alpha,
                pitch_alpha,
                energy_alpha,
                mel_max_len,
                pitch_target,
                energy_target,
                length_target,
            )
            mel_output = self.decoder(mel_output, mel_pos)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_len)
            mel_output = self.mel_linear(mel_output)

            return (
                mel_output,
                duration_predictor_output,
                pitch_predictor_output,
                energy_predictor_output,
            )

        (
            mel_output,
            mel_pos,
            duration_predictor_output,
            pitch_predictor_output,
            energy_predictor_output,
        ) = self.variance_adapter(
            x,
            duration_alpha,
            pitch_alpha,
            energy_alpha,
        )
        mel_output = self.decoder(mel_output, mel_pos)
        mel_output = self.mel_linear(mel_output)

        return mel_output
