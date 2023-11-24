from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import numpy as np


def pad_1D_tensor(inputs, PAD_IDX=0):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD_IDX) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def reprocess_tensor(batch):
    texts = [item["src_seq"] for item in batch]
    mel_targets = [item["mel_target"] for item in batch]
    energy_targets = [item["energy_target"] for item in batch]
    pitch_targets = [item["pitch_target"] for item in batch]
    durations = [item["length_target"] for item in batch]

    length_text = []
    for text in texts:
        length_text.append(text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(
            np.pad(
                [i + 1 for i in range(int(length_src_row))],
                (0, max_len - int(length_src_row)),
                "constant",
            )
        )
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(
            np.pad(
                [i + 1 for i in range(int(length_mel_row))],
                (0, max_mel_len - int(length_mel_row)),
                "constant",
            )
        )
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    energy_targets = pad_1D_tensor(energy_targets)
    pitch_targets = pad_1D_tensor(pitch_targets)

    out = {
        "src_seq": texts,
        "mel_target": mel_targets,
        "energy_target": energy_targets,
        "pitch_target": pitch_targets,
        "length_target": durations,
        "mel_pos": mel_pos,
        "src_pos": src_pos,
        "mel_max_len": max_mel_len,
    }

    return out


def collate_fn(batch: List[dict]):
    len_arr = np.array([d["src_seq"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)

    return reprocess_tensor(batch)
