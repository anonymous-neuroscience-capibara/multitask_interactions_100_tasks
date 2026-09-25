from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class ContextIntegration(CognitiveTask):
    """
    Context-dependent integration: Two stimuli presented sequentially,
    integrate only the relevant modality based on context

    - ContextMod1: integrate modality 1, ignore modality 2
    - ContextMod2: integrate modality 2, ignore modality 1

    This tests: selective attention + integration of noisy evidence
    """

    PHASE_NAMES = ["context", "stimulus1", "stimulus2", "delay", "response"]

    def __init__(self, duration_params: Dict, relevant_modality=1):
        super().__init__(duration_params)
        self.relevant_modality = relevant_modality  # 1 or 2
        self.loss_channels = [1, 2]  # Only evaluate cosine/sine of integrated direction

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(10, 20)  # Short delay
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_stim2 + T_delay + T_response

        # Sample angles and coherences for both modalities
        # Relevant modality has stronger signal
        theta_relevant = np.random.uniform(0, 2 * np.pi)
        coherence_relevant = 0.8

        # Irrelevant modality has different angle, weaker signal
        theta_irrelevant = np.random.uniform(0, 2 * np.pi)
        coherence_irrelevant = 0.3

        # Create inputs: [fixation, mod1_cos, mod1_sin, mod2_cos, mod2_sin, rule_mod1, rule_mod2]
        inputs = torch.zeros(T_total, 7)

        # Context period
        inputs[:T_context, 0] = 1  # fixation
        if self.relevant_modality == 1:
            inputs[:T_context, 5] = 1  # rule_mod1
        else:
            inputs[:T_context, 6] = 1  # rule_mod2

        # First stimulus presentation
        t_stim1_start = T_context
        t_stim1_end = t_stim1_start + T_stim1
        inputs[t_stim1_start:t_stim1_end, 0] = 1

        # Modality 1
        inputs[t_stim1_start:t_stim1_end, 1] = (
            coherence_relevant * np.cos(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.cos(theta_irrelevant)
        )
        inputs[t_stim1_start:t_stim1_end, 2] = (
            coherence_relevant * np.sin(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.sin(theta_irrelevant)
        )

        # Modality 2
        inputs[t_stim1_start:t_stim1_end, 3] = (
            coherence_irrelevant * np.cos(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.cos(theta_relevant)
        )
        inputs[t_stim1_start:t_stim1_end, 4] = (
            coherence_irrelevant * np.sin(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.sin(theta_relevant)
        )

        if self.relevant_modality == 1:
            inputs[t_stim1_start:t_stim1_end, 5] = 1
        else:
            inputs[t_stim1_start:t_stim1_end, 6] = 1

        # Second stimulus presentation (adds more evidence)
        t_stim2_start = t_stim1_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1

        # Same relevant/irrelevant structure
        inputs[t_stim2_start:t_stim2_end, 1] = (
            coherence_relevant * np.cos(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.cos(theta_irrelevant)
        )
        inputs[t_stim2_start:t_stim2_end, 2] = (
            coherence_relevant * np.sin(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.sin(theta_irrelevant)
        )
        inputs[t_stim2_start:t_stim2_end, 3] = (
            coherence_irrelevant * np.cos(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.cos(theta_relevant)
        )
        inputs[t_stim2_start:t_stim2_end, 4] = (
            coherence_irrelevant * np.sin(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.sin(theta_relevant)
        )

        if self.relevant_modality == 1:
            inputs[t_stim2_start:t_stim2_end, 5] = 1
        else:
            inputs[t_stim2_start:t_stim2_end, 6] = 1

        # Delay
        t_delay_start = t_stim2_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.relevant_modality == 1:
            inputs[t_delay_start:t_delay_end, 5] = 1
        else:
            inputs[t_delay_start:t_delay_end, 6] = 1

        # Response period
        t_resp_start = t_delay_end
        if self.relevant_modality == 1:
            inputs[t_resp_start:, 5] = 1
        else:
            inputs[t_resp_start:, 6] = 1

        # Target: respond in direction of relevant modality
        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[1] = np.cos(theta_relevant)
        targets[2] = np.sin(theta_relevant)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(10, 20)
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_stim2 + T_delay + T_response

        theta_relevant = np.random.uniform(0, 2 * np.pi)
        coherence_relevant = 0.8
        theta_irrelevant = np.random.uniform(0, 2 * np.pi)
        coherence_irrelevant = 0.3

        inputs = torch.zeros(T_total, 7)

        inputs[:T_context, 0] = 1
        if self.relevant_modality == 1:
            inputs[:T_context, 5] = 1
        else:
            inputs[:T_context, 6] = 1

        t_stim1_start = T_context
        t_stim1_end = t_stim1_start + T_stim1
        inputs[t_stim1_start:t_stim1_end, 0] = 1
        inputs[t_stim1_start:t_stim1_end, 1] = (
            coherence_relevant * np.cos(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.cos(theta_irrelevant)
        )
        inputs[t_stim1_start:t_stim1_end, 2] = (
            coherence_relevant * np.sin(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.sin(theta_irrelevant)
        )
        inputs[t_stim1_start:t_stim1_end, 3] = (
            coherence_irrelevant * np.cos(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.cos(theta_relevant)
        )
        inputs[t_stim1_start:t_stim1_end, 4] = (
            coherence_irrelevant * np.sin(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.sin(theta_relevant)
        )
        if self.relevant_modality == 1:
            inputs[t_stim1_start:t_stim1_end, 5] = 1
        else:
            inputs[t_stim1_start:t_stim1_end, 6] = 1

        t_stim2_start = t_stim1_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = (
            coherence_relevant * np.cos(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.cos(theta_irrelevant)
        )
        inputs[t_stim2_start:t_stim2_end, 2] = (
            coherence_relevant * np.sin(theta_relevant)
            if self.relevant_modality == 1
            else coherence_irrelevant * np.sin(theta_irrelevant)
        )
        inputs[t_stim2_start:t_stim2_end, 3] = (
            coherence_irrelevant * np.cos(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.cos(theta_relevant)
        )
        inputs[t_stim2_start:t_stim2_end, 4] = (
            coherence_irrelevant * np.sin(theta_irrelevant)
            if self.relevant_modality == 1
            else coherence_relevant * np.sin(theta_relevant)
        )
        if self.relevant_modality == 1:
            inputs[t_stim2_start:t_stim2_end, 5] = 1
        else:
            inputs[t_stim2_start:t_stim2_end, 6] = 1

        t_delay_start = t_stim2_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.relevant_modality == 1:
            inputs[t_delay_start:t_delay_end, 5] = 1
        else:
            inputs[t_delay_start:t_delay_end, 6] = 1

        t_resp_start = t_delay_end
        if self.relevant_modality == 1:
            inputs[t_resp_start:, 5] = 1
        else:
            inputs[t_resp_start:, 6] = 1

        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[1] = np.cos(theta_relevant)
        targets[2] = np.sin(theta_relevant)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_stim1, 1, dtype=np.int64),
                np.full(T_stim2, 2, dtype=np.int64),
                np.full(T_delay, 3, dtype=np.int64),
                np.full(T_response, 4, dtype=np.int64),
            ]
        )

        return inputs, targets, mask, phases
