from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ============================================================================
# Task Definitions
# ============================================================================


class CognitiveTask:
    """Base class for cognitive tasks"""

    def __init__(self, duration_params: Dict):
        self.duration_params = duration_params

    def generate_trial(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: (T, input_dim) - time series inputs
            targets: (output_dim,) - target outputs
            mask: (T,) - which timesteps to include in loss
        """
        raise NotImplementedError


class DelayedResponse(CognitiveTask):
    """
    Delayed Response task: remember stimulus direction and respond after delay
    Pro: respond in same direction as stimulus
    Anti: respond in opposite direction
    """

    def __init__(self, duration_params: Dict, mode="pro"):
        super().__init__(duration_params)
        self.mode = mode  # 'pro' or 'anti'

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_delay = np.random.randint(*self.duration_params["delay"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_delay + T_response

        # Sample stimulus angle
        theta = np.random.uniform(0, 2 * np.pi)

        # Create inputs: [fixation, stim_cos, stim_sin, rule_pro, rule_anti]
        inputs = torch.zeros(T_total, 5)

        # Context period: fixation + rule
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == "pro":
            inputs[:T_context, 3] = 1  # rule_pro
        else:
            inputs[:T_context, 4] = 1  # rule_anti

        # Stimulus period: fixation + stimulus + rule
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1  # fixation
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)  # stim_cos
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)  # stim_sin
        if self.mode == "pro":
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1

        # Delay period: fixation + rule
        t_delay_start = t_stim_end
        t_delay_end = t_delay_start + T_delay
        inputs[t_delay_start:t_delay_end, 0] = 1
        if self.mode == "pro":
            inputs[t_delay_start:t_delay_end, 3] = 1
        else:
            inputs[t_delay_start:t_delay_end, 4] = 1

        # Response period: no fixation, rule still on
        t_resp_start = t_delay_end
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        # Target output: [fixation, response_cos, response_sin]
        targets = torch.zeros(3)
        targets[0] = 0  # no fixation during response
        if self.mode == "pro":
            targets[1] = np.cos(theta)
            targets[2] = np.sin(theta)
        else:  # anti
            targets[1] = np.cos(theta + np.pi)
            targets[2] = np.sin(theta + np.pi)

        # Mask: only evaluate during response period
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask


class ReactionTime(CognitiveTask):
    """
    Reaction time task: respond immediately to stimulus
    Pro: respond in same direction
    Anti: respond in opposite direction
    """

    def __init__(self, duration_params: Dict, mode="pro"):
        super().__init__(duration_params)
        self.mode = mode

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim_response = np.random.randint(*self.duration_params["stimulus"])
        T_total = T_context + T_stim_response

        theta = np.random.uniform(0, 2 * np.pi)

        inputs = torch.zeros(T_total, 5)

        # Context period
        inputs[:T_context, 0] = 1  # fixation
        if self.mode == "pro":
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1

        # Stimulus/Response period: no fixation, stimulus present
        t_resp_start = T_context
        inputs[t_resp_start:, 1] = np.cos(theta)
        inputs[t_resp_start:, 2] = np.sin(theta)
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        targets = torch.zeros(3)
        targets[0] = 0
        if self.mode == "pro":
            targets[1] = np.cos(theta)
            targets[2] = np.sin(theta)
        else:
            targets[1] = np.cos(theta + np.pi)
            targets[2] = np.sin(theta + np.pi)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask


class CategoryDecision(CognitiveTask):
    """
    Category decision task: respond based on whether stimulus is above/below threshold
    Pro: respond in category 1 if theta < pi, category 2 if theta > pi
    Anti: opposite
    """

    def __init__(self, duration_params: Dict, mode="pro"):
        super().__init__(duration_params)
        self.mode = mode

    def generate_trial(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_stim = np.random.randint(*self.duration_params["stimulus"])
        T_response = np.random.randint(*self.duration_params["response"])
        T_total = T_context + T_stim + T_response

        theta = np.random.uniform(0, 2 * np.pi)

        inputs = torch.zeros(T_total, 5)

        # Context
        inputs[:T_context, 0] = 1
        if self.mode == "pro":
            inputs[:T_context, 3] = 1
        else:
            inputs[:T_context, 4] = 1

        # Stimulus
        t_stim_start = T_context
        t_stim_end = t_stim_start + T_stim
        inputs[t_stim_start:t_stim_end, 0] = 1
        inputs[t_stim_start:t_stim_end, 1] = np.cos(theta)
        inputs[t_stim_start:t_stim_end, 2] = np.sin(theta)
        if self.mode == "pro":
            inputs[t_stim_start:t_stim_end, 3] = 1
        else:
            inputs[t_stim_start:t_stim_end, 4] = 1

        # Response
        t_resp_start = t_stim_end
        if self.mode == "pro":
            inputs[t_resp_start:, 3] = 1
        else:
            inputs[t_resp_start:, 4] = 1

        # Determine category
        if self.mode == "pro":
            category_angle = 0 if theta < np.pi else np.pi
        else:  # anti
            category_angle = np.pi if theta < np.pi else 0

        targets = torch.zeros(3)
        targets[0] = 0
        targets[1] = np.cos(category_angle)
        targets[2] = np.sin(category_angle)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets, mask


# ============================================================================
# Dataset
# ============================================================================


class MultiTaskDataset(Dataset):
    """Dataset that generates trials from multiple tasks"""

    def __init__(self, tasks: List[CognitiveTask], n_trials: int = 1000):
        self.tasks = tasks
        self.n_trials = n_trials

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        # Randomly select a task
        task_idx = np.random.randint(len(self.tasks))
        task = self.tasks[task_idx]

        # Generate trial
        inputs, targets, mask = task.generate_trial()

        # Add task identity to inputs (one-hot encoding)
        task_id = torch.zeros(inputs.shape[0], len(self.tasks))
        task_id[:, task_idx] = 1
        inputs = torch.cat([inputs, task_id], dim=1)


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    inputs_list, targets_list, masks_list, task_ids = zip(*batch)

    # Find max length
    max_len = max(inp.shape[0] for inp in inputs_list)
    batch_size = len(batch)
    input_dim = inputs_list[0].shape[1]

    # Pad sequences
    inputs_padded = torch.zeros(batch_size, max_len, input_dim)
    masks_padded = torch.zeros(batch_size, max_len)

    for i, (inp, mask) in enumerate(zip(inputs_list, masks_list)):
        T = inp.shape[0]
        inputs_padded[i, :T] = inp
        masks_padded[i, :T] = mask

    targets = torch.stack(targets_list)
    task_ids = torch.tensor(task_ids)

    return inputs_padded, targets, masks_padded, task_ids
