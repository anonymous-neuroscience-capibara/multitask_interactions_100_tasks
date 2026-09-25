from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class MultiSensoryIntegration(CognitiveTask):
    """
    Multi-sensory Integration task: integrate two simultaneous stimuli.

    Timeline:
    - Context: fixation only
    - Stimulus: two modalities shown simultaneously, each pointing to a
      direction with some coherence (strength)
    - Delay: hold integrated estimate
    - Response: report the combined direction

    The correct response is the direction of the vector sum of both stimuli.
    Each modality has its own coherence, so the network must weight them
    appropriately.

    Unlike ContextIntegration, there is no rule channel — the network
    must always integrate both modalities equally.

    Output: angular response (fixation, cos, sin)

    Input channels (5): [fixation, mod1_cos, mod1_sin, mod2_cos, mod2_sin]

    Tests: simultaneous evidence integration + vector addition + working memory

    Reference: Raposo et al. (2014), Multisensory decision-making in rats
    and humans
    """

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(
        self,
        duration_params: Dict,
        sigma: float = 0.1,
    ):
        super().__init__(duration_params)
        self.sigma = sigma
        self.is_binary_task = False
        self.loss_channels = [1, 2]  # evaluate cosine and sine separately

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        # Sample directions and coherences for both modalities
        theta1 = np.random.uniform(0, 2 * np.pi)
        theta2 = np.random.uniform(0, 2 * np.pi)
        coherence1 = np.random.uniform(0.3, 1.0)
        coherence2 = np.random.uniform(0.3, 1.0)

        # Correct response: direction of vector sum
        sum_x = coherence1 * np.cos(theta1) + coherence2 * np.cos(theta2)
        sum_y = coherence1 * np.sin(theta1) + coherence2 * np.sin(theta2)
        # Normalize to unit vector (if nonzero)
        magnitude = np.sqrt(sum_x**2 + sum_y**2)
        if magnitude > 1e-6:
            target_cos = sum_x / magnitude
            target_sin = sum_y / magnitude
        else:
            # Degenerate case: opposite directions cancel out
            target_cos = 0.0
            target_sin = 0.0

        # Create inputs: [fixation, mod1_cos, mod1_sin, mod2_cos, mod2_sin]
        inputs = torch.zeros(T_total, 5)

        # Context period
        inputs[:T_context, 0] = 1

        # Stimulus period: both modalities simultaneously + noise
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        noise = torch.randn(T_stim, 4) * self.sigma
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = coherence1 * np.cos(theta1) + noise[:, 0]
        inputs[t_stim_start:t_stim_end, 2] = coherence1 * np.sin(theta1) + noise[:, 1]
        inputs[t_stim_start:t_stim_end, 3] = coherence2 * np.cos(theta2) + noise[:, 2]
        inputs[t_stim_start:t_stim_end, 4] = coherence2 * np.sin(theta2) + noise[:, 3]

        # Delay period: fixation only
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1

        # Response period: no fixation
        t_resp_start = t_delay_end

        # Target: angular response
        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0  # no fixation
        targets[1] = target_cos
        targets[2] = target_sin

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        theta1 = np.random.uniform(0, 2 * np.pi)
        theta2 = np.random.uniform(0, 2 * np.pi)
        coherence1 = np.random.uniform(0.3, 1.0)
        coherence2 = np.random.uniform(0.3, 1.0)

        sum_x = coherence1 * np.cos(theta1) + coherence2 * np.cos(theta2)
        sum_y = coherence1 * np.sin(theta1) + coherence2 * np.sin(theta2)
        magnitude = np.sqrt(sum_x**2 + sum_y**2)
        if magnitude > 1e-6:
            target_cos = sum_x / magnitude
            target_sin = sum_y / magnitude
        else:
            target_cos = 0.0
            target_sin = 0.0

        inputs = torch.zeros(T_total, 5)

        inputs[:T_context, 0] = 1

        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        noise = torch.randn(T_stim, 4) * self.sigma
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = coherence1 * np.cos(theta1) + noise[:, 0]
        inputs[t_stim_start:t_stim_end, 2] = coherence1 * np.sin(theta1) + noise[:, 1]
        inputs[t_stim_start:t_stim_end, 3] = coherence2 * np.cos(theta2) + noise[:, 2]
        inputs[t_stim_start:t_stim_end, 4] = coherence2 * np.sin(theta2) + noise[:, 3]

        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1

        t_resp_start = t_delay_end

        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0
        targets[1] = target_cos
        targets[2] = target_sin

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim, 1, dtype=np.int64),
                np.full(T_delay, 2, dtype=np.int64),
                np.full(T_response, 3, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
