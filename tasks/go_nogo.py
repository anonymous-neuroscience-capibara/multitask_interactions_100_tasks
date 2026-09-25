from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class GoNogo(CognitiveTask):
    """
    Go/No-go task with delay: threshold decision on stimulus amplitude.

    Timeline:
    - Context: fixation only
    - Stimulus: random direction with varying amplitude
    - Delay: hold decision in memory
    - Response: go (output direction 0°) if amplitude > threshold, maintain fixation if not

    Tests: threshold detection + response inhibition + working memory for decision

    Input channels (4): [fixation, stim_cos, stim_sin, rule]
    """

    PHASE_NAMES = ["context", "stimulus", "delay", "response"]

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.threshold = 0.5
        self.is_binary_task = True
        self.loss_channels = [
            self.BINARY_CHANNEL
        ]  # go/nogo decision (ch0 always trained)

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = 0
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        theta = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.2, 1.0)
        is_go = amplitude > self.threshold

        # Create inputs: [fixation, stim_cos, stim_sin, rule]
        inputs = torch.zeros(T_total, 4)

        # Context period
        inputs[:T_context, 0] = 1
        inputs[:T_context, 3] = 1

        # Stimulus period
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = amplitude * np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = amplitude * np.sin(theta)
        inputs[t_stim_start:t_stim_end, 3] = 1

        # Delay period
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        inputs[t_delay_start:t_delay_end, 3] = 1

        # Response period
        t_resp_start = t_delay_end
        inputs[t_resp_start:, 3] = 1

        # Target
        targets = torch.zeros(self.output_dim)
        if is_go:
            targets[0] = 0.0  # break fixation
            targets[self.BINARY_CHANNEL] = 1.0
        else:
            targets[0] = 1.0  # maintain fixation
            targets[self.BINARY_CHANNEL] = 0.0

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = 0
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        theta = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.2, 1.0)
        is_go = amplitude > self.threshold

        inputs = torch.zeros(T_total, 4)

        inputs[:T_context, 0] = 1
        inputs[:T_context, 3] = 1

        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = amplitude * np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = amplitude * np.sin(theta)
        inputs[t_stim_start:t_stim_end, 3] = 1

        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        inputs[t_delay_start:t_delay_end, 3] = 1

        t_resp_start = t_delay_end
        inputs[t_resp_start:, 3] = 1

        targets = torch.zeros(self.output_dim)
        if is_go:
            targets[0] = 0.0
            targets[self.BINARY_CHANNEL] = 1.0
        else:
            targets[0] = 1.0
            targets[self.BINARY_CHANNEL] = 0.0

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phase_parts = [
            np.full(T_context, 0, dtype=np.int64),
            np.full(T_stim, 1, dtype=np.int64),
        ]
        if T_delay > 0:
            phase_parts.append(np.full(T_delay, 2, dtype=np.int64))
        phase_parts.append(np.full(T_response, 3, dtype=np.int64))
        phases = np.concatenate(phase_parts)

        return inputs, targets, mask, phases


# class old_GoNogo(CognitiveTask):
#     """
#     Go/Nogo task: Simple decision task
#     - Go: respond in a fixed direction when stimulus amplitude > threshold
#     - Nogo: maintain fixation when stimulus amplitude < threshold

#     This tests: simple threshold decision + motor control
#     """
#     def __init__(self, duration_params: Dict):
#         super().__init__(duration_params)
#         self.threshold = 0.5  # Amplitude threshold
#         self.loss_channels = [1,2]

#     def generate_trial(self):
#         T_context = np.random.randint(*self.duration_params['context'])
#         T_stim = np.random.randint(*self.duration_params['stimulus'])
#         T_response = np.random.randint(*self.duration_params['response'])
#         T_total = T_context + T_stim + T_response

#         # Random angle and amplitude
#         theta = np.random.uniform(0, 2*np.pi)
#         amplitude = np.random.uniform(0.2, 1.0)
#         is_go = amplitude > self.threshold

#         # Create inputs: [fixation, stim_cos, stim_sin, rule_gonogo]
#         inputs = torch.zeros(T_total, 4)

#         # Context period
#         inputs[:T_context, 0] = 1
#         inputs[:T_context, 3] = 1  # rule

#         # Stimulus period
#         t_stim_start = T_context
#         t_stim_end = t_stim_start + T_stim
#         inputs[t_stim_start:t_stim_end, 0] = 1
#         inputs[t_stim_start:t_stim_end, 1] = amplitude * np.cos(theta)
#         inputs[t_stim_start:t_stim_end, 2] = amplitude * np.sin(theta)
#         inputs[t_stim_start:t_stim_end, 3] = 1

#         # Response period
#         t_resp_start = t_stim_end
#         inputs[t_resp_start:, 3] = 1

#         # Target: if go, respond in fixed direction (0°); if nogo, maintain fixation
#         targets = torch.zeros(self.output_dim)
#         if is_go:
#             targets[0] = 0  # break fixation
#             targets[1] = 1.0  # cos(0)
#             targets[2] = 0.0  # sin(0)
#         else:
#             targets[0] = 1  # maintain fixation
#             targets[1] = 0.0
#             targets[2] = 0.0

#         mask = torch.zeros(T_total)
#         mask[t_resp_start:] = 1

#         return inputs, targets, mask
