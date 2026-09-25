from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class DelayedResponse32D(CognitiveTask):
    """
    Delayed Response task with population-coded stimulus and output.

    Timeline:
    - Context: fixation only
    - Stimulus: population-coded direction presented
    - Delay: must hold direction in memory
    - Response: reproduce direction as population code

    Input channels (1 + N_NEURONS = 33 by default):
        [fixation, pop_0, pop_1, ..., pop_N-1]

    Target channels (1 + N_NEURONS = 33 by default):
        [fixation, pop_0, pop_1, ..., pop_N-1]

    Args:
        duration_params : dict with keys "context", "stimulus", "delay",
                          "response", each a (min, max) tuple in timesteps.
        mode            : "pro"  → respond toward stimulus
                          "anti" → respond away from stimulus (θ + π)
        n_neurons       : number of direction-tuned neurons (default 32)
    """

    def __init__(
        self,
        duration_params: Dict,
        mode: str = "pro",
        n_neurons: int = 32,
    ):
        super().__init__(duration_params)
        assert mode in ("pro", "anti"), "mode must be 'pro' or 'anti'"
        self.mode = mode
        self.n_neurons = n_neurons
        self.preferred_angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
        self.n_input = 1 + n_neurons
        self.output_dim = 1 + n_neurons
        self.is_population_task = True

    def _population_code(self, angle: float) -> np.ndarray:
        return np.cos(angle - self.preferred_angles)  # (n_neurons,)

    def generate_trial(self):
        # Sample durations from duration_params
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        # Sample stimulus angle
        theta = np.random.uniform(0, 2 * np.pi)
        response_angle = theta if self.mode == "pro" else theta + np.pi

        # Create inputs: [fixation, pop_0, ..., pop_N-1]
        inputs = torch.zeros(T_total, self.n_input)

        # Context period: fixation only
        inputs[:T_context, 0] = 1  # fixation

        # Stimulus period: fixation + population code
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1  # fixation
        inputs[t_stim_start:t_stim_end, 1:] = torch.tensor(
            self._population_code(theta), dtype=torch.float32
        )

        # Delay period: fixation only, no stimulus
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1  # fixation

        # Response period: no fixation
        t_resp_start = t_delay_end

        # Target: [fixation=0, population code of response angle]
        targets = torch.zeros(self.output_dim)
        targets[0] = 0.0  # break fixation
        targets[1:] = torch.tensor(
            self._population_code(response_angle), dtype=torch.float32
        )

        # Mask: evaluate at response onset
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask
