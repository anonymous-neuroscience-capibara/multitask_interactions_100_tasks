from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class NoiseCleaner(CognitiveTask):
    """
    Filter noisy samples to estimate true direction
    """

    PHASE_NAMES = ["context", "stimulus", "response"]

    def __init__(self, duration_params: Dict, noise_level: float = 0.5):
        super().__init__(duration_params)
        self.noise_level = noise_level
        self.loss_channels = [1, 2]  # Only evaluate cosine/sine of target direction

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        theta_true = np.random.uniform(0, 2 * np.pi)

        # Use NumPy array first
        inputs_np = np.zeros((T_total, 3))

        # Context
        inputs_np[:T_context, 0] = 1

        # Stimulus: noisy samples
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim

        noise = np.random.uniform(-self.noise_level, self.noise_level, size=T_stim)
        theta_noisy = theta_true + noise

        inputs_np[t_stim_start:t_stim_end, 0] = 1
        inputs_np[t_stim_start:t_stim_end, 1] = np.cos(theta_noisy)
        inputs_np[t_stim_start:t_stim_end, 2] = np.sin(theta_noisy)

        # Convert to torch at end
        inputs = torch.from_numpy(inputs_np).float()

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[1] = np.cos(theta_true)
        targets[2] = np.sin(theta_true)

        # Response starts after stimulus
        t_resp_start = t_stim_end
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        theta_true = np.random.uniform(0, 2 * np.pi)

        inputs_np = np.zeros((T_total, 3))

        inputs_np[:T_context, 0] = 1

        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim

        noise = np.random.uniform(-self.noise_level, self.noise_level, size=T_stim)
        theta_noisy = theta_true + noise

        inputs_np[t_stim_start:t_stim_end, 0] = 1
        inputs_np[t_stim_start:t_stim_end, 1] = np.cos(theta_noisy)
        inputs_np[t_stim_start:t_stim_end, 2] = np.sin(theta_noisy)

        inputs = torch.from_numpy(inputs_np).float()

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[1] = np.cos(theta_true)
        targets[2] = np.sin(theta_true)

        t_resp_start = t_stim_end
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim, 1, dtype=np.int64),
                np.full(T_response, 2, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
