from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class SoundLocalization(CognitiveTask):
    """
    Auditory azimuth localization (Jeffress 1948 place code; Knudsen & Konishi
    1978, barn owl ICx space map).

    Two 'ear' channels carry a noisy sound whose left/right level difference
    (ILD) encodes the source azimuth; the network must report the azimuth as a
    scalar in [-1, 1] (-1 = full left, +1 = full right). The cue is noisy and
    spread over a short window, so a robust estimate requires averaging. Adds an
    auditory spatial-mapping modality the battery otherwise lacks; the readout
    is a scalar (magnitude), not a ring direction.

    Input channels (3): [fixation, left_ear, right_ear].
    Output: channel 0 = fixation; channel 3 (scalar) = azimuth.
    """

    PHASE_NAMES = ["context", "sound", "response"]
    is_scalar_task = True

    def __init__(self, duration_params: Dict, noise: float = 0.15):
        super().__init__(duration_params)
        self.noise = noise
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_sound = np.random.randint(*dp["stimulus"])
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_sound + T_response

        az = float(np.random.uniform(-1.0, 1.0))
        left_level = (1.0 - az) / 2.0   # az=-1 -> all left
        right_level = (1.0 + az) / 2.0  # az=+1 -> all right

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        nl = np.random.normal(0, self.noise, size=T_sound)
        nr = np.random.normal(0, self.noise, size=T_sound)
        inputs[T_context:T_context + T_sound, 1] = torch.from_numpy(
            (left_level + nl).astype(np.float32))
        inputs[T_context:T_context + T_sound, 2] = torch.from_numpy(
            (right_level + nr).astype(np.float32))
        t_resp = T_context + T_sound
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = az  # scalar azimuth readout

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_sound, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
