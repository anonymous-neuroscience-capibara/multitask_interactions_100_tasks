from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class PredictiveTracking(CognitiveTask):
    # sembla que l'accuracy baixa quan l'afegeixo

    """
    Predictive Tracking task: Predict future position of moving target after occlusion

    Timeline:
    - Context period
    - Visible motion: Target moves at constant velocity
    - Occlusion: Target disappears - must extrapolate
    - Response: Report predicted position

    Physics: θ(t) = θ₀ + ω*t (constant velocity)

    Tests: Motion extrapolation + temporal prediction + velocity estimation
    """

    def __init__(self, duration_params: Dict):
        super().__init__(duration_params)
        self.loss_channels = [1, 2]  # Only evaluate cosine/sine of predicted position

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_visible = np.random.randint(*self.duration_params["stimulus"])
        T_occlusion = 10
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_visible + T_occlusion + T_response

        # Sample initial position and velocity
        theta_0 = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(-0.05, 0.05)  # Constant angular velocity

        # Create inputs: [fixation, target_cos, target_sin]
        inputs = torch.zeros(T_total, 3)

        # Context period
        inputs[:T_context, 0] = 1  # fixation

        # Visible motion phase
        t_visible_start = T_context
        t_visible_end = t_visible_start + T_visible

        for t in range(T_visible):
            # Calculate position: θ(t) = θ₀ + ω*t
            theta_t = theta_0 + omega * t
            theta_t = theta_t % (2 * np.pi)

            # Show moving target
            t_actual = t_visible_start + t
            inputs[t_actual, 0] = 1  # fixation
            inputs[t_actual, 1] = np.cos(theta_t)
            inputs[t_actual, 2] = np.sin(theta_t)

        # Occlusion phase - target disappears
        t_occlusion_start = t_visible_end
        t_occlusion_end = t_occlusion_start + T_occlusion
        inputs[t_occlusion_start:t_occlusion_end, 0] = 1  # fixation only

        # Calculate predicted position at end of occlusion
        t_predict = T_visible + T_occlusion
        theta_predicted = theta_0 + omega * t_predict
        theta_predicted = theta_predicted % (2 * np.pi)

        # Target: predicted position
        targets = torch.zeros(self.output_dim)
        targets[0] = 0
        targets[1] = np.cos(theta_predicted)
        targets[2] = np.sin(theta_predicted)

        # Response starts after occlusion
        t_resp_start = t_occlusion_end
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask
