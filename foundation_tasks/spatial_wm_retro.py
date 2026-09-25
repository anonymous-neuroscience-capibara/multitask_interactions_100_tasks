from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class SpatialWMRetro(CognitiveTask):
    """
    Spatial working memory with a retro-cue (selection from memory).

    Two location cues are presented in sequence, held over a delay, then a
    retro-cue indicates *which* of the two to report; the network outputs the
    selected location. Requires maintaining two items simultaneously and gating
    one out at report time -- a stronger WM demand than single-item delay
    (Funahashi 1989 ODR + retro-cue selection, e.g. Griffin & Nobre 2003).

    Input channels (4): [fixation, cue_cos, cue_sin, retrocue].
                        retrocue = 0 -> report cue1, 1 -> report cue2.
    Output: channel 0 = fixation; channels 1,2 = (cos,sin) of selected angle.
    Accuracy: angular (default branch, last masked timestep).
    """

    PHASE_NAMES = ["context", "cue1", "cue2", "delay", "retrocue", "response"]

    def __init__(self, duration_params: Dict, n_angles: int = 8):
        super().__init__(duration_params)
        self.n_angles = n_angles
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_cue = 2
        T_delay = np.random.randint(*dp["delay"])
        T_retro = 2
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + 2 * T_cue + T_delay + T_retro + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        a1, a2 = np.random.choice(self.n_angles, size=2, replace=False)
        report_second = np.random.rand() < 0.5
        sel = a2 if report_second else a1

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        t1 = T_context
        inputs[t1:t1 + T_cue, 1] = np.cos(angles[a1])
        inputs[t1:t1 + T_cue, 2] = np.sin(angles[a1])
        t2 = t1 + T_cue
        inputs[t2:t2 + T_cue, 1] = np.cos(angles[a2])
        inputs[t2:t2 + T_cue, 2] = np.sin(angles[a2])
        t_retro = t2 + T_cue + T_delay
        inputs[t_retro:t_retro + T_retro, 3] = 1.0 if report_second else -1.0
        t_resp = t_retro + T_retro
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(angles[sel])
        targets[2] = np.sin(angles[sel])

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_cue, T_cue, T_delay, T_retro, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tq1, Tq2, Td, Tre, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tq1, 1, dtype=np.int64),
            np.full(Tq2, 2, dtype=np.int64), np.full(Td, 3, dtype=np.int64),
            np.full(Tre, 4, dtype=np.int64), np.full(Tr, 5, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
