from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class RTResponse(CognitiveTask):
    """
    Reaction-time Go / Anti (no delay; respond as soon as the stimulus appears).

    Pro: respond toward the stimulus. Anti: respond opposite (antisaccade,
    Munoz & Everling 2004). Unlike the delayed versions there is no memory gap,
    so these are near-reactive -- a useful low-complexity anchor (`degrees`).

    Input channels (3): [fixation, stim_cos, stim_sin].
    Output: channel 0 = fixation; channels 1,2 = (cos,sin) of response.
    """

    PHASE_NAMES = ["context", "response"]
    # Reaction task: score accuracy right after stimulus onset (the *reaction*),
    # not at the end of the response window. `reaction_step` is 0-indexed into
    # the masked timesteps; 1 = the 2nd response step, giving the RNN one step to
    # propagate the stimulus into its state before we read the response.
    is_reaction_task = True
    reaction_step = 1

    def __init__(self, duration_params: Dict, mode="pro"):
        super().__init__(duration_params)
        self.mode = mode  # 'pro' or 'anti'
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_response = np.random.randint(*dp["stimulus"])  # respond while stimulus is on
        T_total = T_context + T_response

        theta = np.random.uniform(0, 2 * np.pi)
        resp = theta if self.mode == "pro" else (theta + np.pi)

        inputs = torch.zeros(T_total, 3)
        inputs[:T_context, 0] = 1.0  # fixation during context
        inputs[T_context:, 1] = np.cos(theta)  # stimulus stays on through response
        inputs[T_context:, 2] = np.sin(theta)

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(resp)
        targets[2] = np.sin(resp)

        mask = torch.zeros(T_total)
        mask[T_context:] = 1.0
        return inputs, targets, mask, (T_context, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Tr, 1, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
