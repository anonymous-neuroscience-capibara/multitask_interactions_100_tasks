from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


# class Arithmetics(CognitiveTask):
#     """
#     Arithmetic task (1D): perform arithmetic on two scalar values based on rule.

#     Timeline:
#     - Context: fixation + rule
#     - Stimulus 1: first scalar value
#     - Stimulus 2: second scalar value
#     - Response: output the result

#     Modes:
#     - add: a + b
#     - multiply: a * b

#     Values sampled from [0, 1]. Output on last dimension.

#     Input channels (4): [fixation, stimulus, rule_add, rule_mul]

#     Tests: compositional computation + rule-switching
#     """

#     def __init__(self, duration_params: Dict, mode="add"):
#         super().__init__(duration_params)
#         self.mode = mode  # 'add' or 'multiply'
#         self.is_scalar_task = True
#         self.loss_channels = [self.BINARY_CHANNEL]  # Only evaluate last channel

#     def generate_trial(self):
#         T_context = np.random.randint(*self.duration_params["context"])
#         T_stim1 = np.random.randint(*self.duration_params["stimulus"])
#         T_stim2 = np.random.randint(*self.duration_params["stimulus"])
#         T_response = np.random.randint(*self.duration_params["response"])
#         T_total = T_context + T_stim1 + T_stim2 + T_response

#         # Sample two scalar values in [0, 1]
#         a = np.random.uniform(0, 1)
#         b = np.random.uniform(0, 1)

#         if self.mode == "add":
#             result = a + b  # range [0, 2]
#             rule_channel = 2
#         elif self.mode == "multiply":
#             result = a * b  # range [0, 1]
#             rule_channel = 3

#         # Create inputs: [fixation, stimulus, rule_add, rule_mul]
#         inputs = torch.zeros(T_total, 4)
#         inputs[:, rule_channel] = 1  # rule throughout trial

#         # Context period
#         inputs[:T_context, 0] = 1  # fixation

#         # First stimulus
#         t_stim_start = T_context
#         t_stim_end = t_stim_start + T_stim1
#         inputs[t_stim_start:t_stim_end, 0] = 1
#         inputs[t_stim_start:t_stim_end, 1] = a

#         # Second stimulus (same channel, sequential)
#         t_stim2_start = t_stim_end
#         t_stim2_end = t_stim2_start + T_stim2
#         inputs[t_stim2_start:t_stim2_end, 0] = 1
#         inputs[t_stim2_start:t_stim2_end, 1] = b

#         # Response period: no fixation
#         t_resp_start = t_stim2_end

#         # Target: result on last dimension
#         targets = torch.zeros(self.output_dim)
#         targets[0] = 0
#         targets[self.BINARY_CHANNEL] = result

#         # Mask: evaluate at response onset
#         mask = torch.zeros(T_total)
#         mask[t_resp_start:] = 1

#         return inputs, targets, mask


class ComplexProd(CognitiveTask):
    """
    Complex multiplication: given two angles θ1, θ2, compute e^{iθ1} · e^{iθ2} = e^{i(θ1+θ2)}

    Timeline:
    - Context: fixation
    - Stimulus 1: first angle (cos, sin)
    - Stimulus 2: second angle (cos, sin)
    - Response: output the product angle

    Input channels (3): [fixation, cos, sin]
    Output channels (3): [fixation, cos, sin]
    """

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.is_angle_task = True
        self.loss_channels = [1, 2]

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim1 = np.random.randint(*self.duration_params["stimulus"])
        T_stim2 = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim1 + T_stim2 + T_response

        theta1 = np.random.uniform(0, 2 * np.pi)
        theta2 = np.random.uniform(0, 2 * np.pi)

        # e^{iθ1} * e^{iθ2} = e^{i(θ1+θ2)}
        result_cos = np.cos(theta1 + theta2)
        result_sin = np.sin(theta1 + theta2)

        # Inputs: [fixation, cos, sin]
        inputs = torch.zeros(T_total, 3)

        # Context period
        inputs[:T_context, 0] = 1

        # First stimulus
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim1
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta1)
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta1)

        # Second stimulus
        t_stim2_start = t_stim_end
        t_stim2_end = t_stim2_start + T_stim2
        inputs[t_stim2_start:t_stim2_end, 0] = 1
        inputs[t_stim2_start:t_stim2_end, 1] = np.cos(theta2)
        inputs[t_stim2_start:t_stim2_end, 2] = np.sin(theta2)

        # Targets
        t_resp_start = t_stim2_end
        targets = torch.zeros(T_total, self.output_dim)
        targets[t_resp_start:, 1] = result_cos
        targets[t_resp_start:, 2] = result_sin

        # Mask
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask
