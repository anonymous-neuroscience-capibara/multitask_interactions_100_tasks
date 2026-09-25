from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class HeadDirection(CognitiveTask):
    """
    Head-direction integration / ring attractor (Kim, Rouault, Druckmann &
    Jayaraman 2017; Cueva & Wei 2018).

    An initial heading is cued, then a stream of angular velocities is provided;
    the network must report the running heading (integral of angular velocity)
    as a point on the ring at every step. A correct solution is a continuous
    (ring) attractor -- a knife's-edge dynamical structure that gradient descent
    may struggle to find, making this a strong `**` probe.

    Input channels (4): [fixation, init_cos, init_sin, angular_velocity].
                        (init heading shown as cos/sin on ch1,2 during context;
                         angular velocity on ch3 during integration.)
    Output channels: channel 0 = fixation; channels 1,2 = (cos,sin) of heading.
    Accuracy reuses the per-timestep angular metric (is_copytask branch).
    """

    PHASE_NAMES = ["context", "integration"]
    is_copytask = True  # per-timestep angular accuracy

    def __init__(self, duration_params: Dict, omega_max: float = 0.3,
                 int_len_range=(20, 35)):
        super().__init__(duration_params)
        self.omega_max = omega_max
        self.int_len_range = int_len_range
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_int = int(np.random.randint(*self.int_len_range))
        T_total = T_context + T_int

        h0 = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(-self.omega_max, self.omega_max, size=T_int)
        heading = h0 + np.cumsum(omega)

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        inputs[:T_context, 1] = np.cos(h0)  # initial heading cue
        inputs[:T_context, 2] = np.sin(h0)
        inputs[T_context:, 3] = torch.from_numpy(omega.astype(np.float32))  # ang. velocity

        targets = torch.zeros(T_total, self.output_dim)
        targets[T_context:, 1] = torch.from_numpy(np.cos(heading).astype(np.float32))
        targets[T_context:, 2] = torch.from_numpy(np.sin(heading).astype(np.float32))

        mask = torch.zeros(T_total)
        mask[T_context:] = 1.0
        return inputs, targets, mask, (T_context, T_int)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ti) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ti, 1, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
