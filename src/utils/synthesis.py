import numpy as np
import torch

import waveglow
import text as text_module
import audio
import utils

WaveGlow = utils.get_WaveGlow()
WaveGlow = WaveGlow.cuda()


def synthesis(model, text, output_path, device, alpha=1.0):
    text = [text_module.text_to_sequence(text, ["english_cleaners"])]
    text = np.array(text)
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha).contiguous().transpose(1, 2)

    waveglow.inference.inference(mel, WaveGlow, f"{output_path}.wav")
