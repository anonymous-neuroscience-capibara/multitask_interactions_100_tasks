from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class PathIntegration2D(CognitiveTask):
    """
    2D path integration / grid-cell navigation (Cueva & Wei 2018; Banino et al.
    2018; Gardner et al. 2022 toroidal grid code).

    A stream of 2D velocities (vx, vy) is provided; the network integrates them
    into a running (x, y) position, then must report the final position after
    the stream ends (held through a response gap with no input). A correct
    solution maintains a 2D continuous (plane / toroidal) attractor that both
    integrates and stores -- the 2D counterpart of PathIntegration1D's line
    attractor, and the strongest continuous-attractor probe in the battery.

    Input channels (3): [fixation, vx, vy].
    Output: channels 3, 4 (scalar readouts) = final (x, y) position.
    Accuracy: vector-scalar metric -- correct if BOTH coordinates are within
    NON_ANGULAR_THRESHOLD of the target at the last masked timestep.
    """

    PHASE_NAMES = ["context", "integration", "response"]
    is_vector_task = True  # multi-channel scalar readout (see rnn_model.py)

    def __init__(self, duration_params: Dict, vmax: float = 0.1,
                 int_len_range=(15, 30)):
        super().__init__(duration_params)
        self.vmax = vmax
        self.int_len_range = int_len_range
        self.loss_channels = [3, 4]  # (x, y) on channels 3 and 4

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_int = int(np.random.randint(*self.int_len_range))
        T_response = np.random.randint(*dp["response"])
        T_total = T_context + T_int + T_response

        vx = np.random.uniform(-self.vmax, self.vmax, size=T_int)
        vy = np.random.uniform(-self.vmax, self.vmax, size=T_int)
        x = float(np.sum(vx))
        y = float(np.sum(vy))

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        inputs[T_context:T_context + T_int, 1] = torch.from_numpy(vx.astype(np.float32))
        inputs[T_context:T_context + T_int, 2] = torch.from_numpy(vy.astype(np.float32))
        t_resp = T_context + T_int
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[3] = x
        targets[4] = y

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_int, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ti, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ti, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
