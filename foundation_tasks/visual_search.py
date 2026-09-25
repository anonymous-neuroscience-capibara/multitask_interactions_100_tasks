from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class VisualSearch(CognitiveTask):
    """
    Visual search (Treisman & Gelade, 1980).

    A target feature (a direction on the ring) is cued; then an array of
    `set_size` items is presented one per step. The network reports whether the
    cued target is *present* in the array (1) or *absent* (0). Search load
    scales with set_size. Tests attentional selection of a target among
    distractors -- the attention domain the rest of the battery lacks.

    Distinct from Sternberg (which holds a *set* then tests one probe): here a
    single target is cued first and the network scans an array for it
    (prospective search vs. retrospective recognition).

    Input channels (4): [fixation, item_cos, item_sin, target_cue_flag].
                        target_cue_flag = 1 while the cued target is shown.
    Output: channel 0 = fixation; channel 3 = target-present decision (binary).
    """

    PHASE_NAMES = ["context", "cue", "array", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, set_size: int = 4, n_angles: int = 8):
        super().__init__(duration_params)
        self.set_size = set_size
        self.n_angles = n_angles
        self.loss_channels = [self.BINARY_CHANNEL]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_cue = 2
        T_item = 2
        T_response = np.random.randint(*dp["response"])
        T_arr = self.set_size * T_item
        T_total = T_context + T_cue + T_arr + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        target = np.random.randint(self.n_angles)
        present = np.random.rand() < 0.5
        if present:
            arr = list(np.random.randint(0, self.n_angles, size=self.set_size))
            arr[np.random.randint(self.set_size)] = target  # plant the target
        else:
            distractors = [a for a in range(self.n_angles) if a != target]
            arr = list(np.random.choice(distractors, size=self.set_size))

        inputs = torch.zeros(T_total, 4)
        inputs[:, 0] = 1.0
        # Cue the target.
        t_cue = T_context
        inputs[t_cue:t_cue + T_cue, 1] = np.cos(angles[target])
        inputs[t_cue:t_cue + T_cue, 2] = np.sin(angles[target])
        inputs[t_cue:t_cue + T_cue, 3] = 1.0  # target-cue flag
        # Present the array.
        t_arr = t_cue + T_cue
        for i, a in enumerate(arr):
            t0 = t_arr + i * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[a])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[a])
        t_resp = t_arr + T_arr
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if present else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_cue, T_arr, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tcue, Ta, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tcue, 1, dtype=np.int64),
            np.full(Ta, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
