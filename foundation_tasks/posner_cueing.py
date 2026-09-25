from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class PosnerCueing(CognitiveTask):
    """
    Posner spatial cueing / covert attention (Posner, 1980).

    A spatial cue indicates the likely location (direction) of an upcoming
    target. The target then appears -- at the cued direction on 'valid' trials
    (the majority) or elsewhere on 'invalid' trials -- shown weakly and noisily.
    The network reports the target's true direction. Because the cue is a valid
    prior on most trials, *using* it denoises the report on valid trials (and
    mildly hurts on invalid ones) -- the supervised analogue of Posner's
    validity benefit / cost. Tests use of a spatial prior to guide perception.

    Input channels (4): [fixation, dir_cos, dir_sin, target_flag].
                        dir = cue direction during the cue phase, then the noisy
                        target during the target phase; target_flag marks the
                        target phase.
    Output: channel 0 = fixation; channels 1, 2 = (cos, sin) of the TRUE target.
    Accuracy: angular (last masked timestep, within threshold).
    """

    PHASE_NAMES = ["context", "cue", "delay", "target", "response"]

    def __init__(self, duration_params: Dict, n_angles: int = 8,
                 validity: float = 0.75, target_amp: float = 0.6,
                 noise: float = 0.25):
        super().__init__(duration_params)
        self.n_angles = n_angles
        self.validity = validity
        self.target_amp = target_amp
        self.noise = noise
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_cue = 2
        T_delay = np.random.randint(*dp["delay"])
        T_target = 3
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_cue + T_delay + T_target + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        cue = np.random.randint(self.n_angles)
        valid = np.random.rand() < self.validity
        if valid:
            target = cue
        else:
            target = np.random.choice([a for a in range(self.n_angles) if a != cue])

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        # Cue: clean direction, no target flag.
        t_cue = T_context
        inputs[t_cue:t_cue + T_cue, 1] = np.cos(angles[cue])
        inputs[t_cue:t_cue + T_cue, 2] = np.sin(angles[cue])
        # Target: weak + noisy direction, target flag on.
        t_tar = T_context + T_cue + T_delay
        noise_c = np.random.normal(0, self.noise, size=T_target)
        noise_s = np.random.normal(0, self.noise, size=T_target)
        inputs[t_tar:t_tar + T_target, 1] = torch.from_numpy(
            (self.target_amp * np.cos(angles[target]) + noise_c).astype(np.float32))
        inputs[t_tar:t_tar + T_target, 2] = torch.from_numpy(
            (self.target_amp * np.sin(angles[target]) + noise_s).astype(np.float32))
        inputs[t_tar:t_tar + T_target, 3] = 1.0  # target-phase flag
        t_resp = t_tar + T_target
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(angles[target])  # report the TRUE (clean) target dir
        targets[2] = np.sin(angles[target])

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_cue, T_delay, T_target, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tcue, Td, Tt, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tcue, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tt, 3, dtype=np.int64),
            np.full(Tr, 4, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
