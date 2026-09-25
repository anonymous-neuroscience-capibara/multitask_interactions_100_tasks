from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CognitiveTask:
    """Base class for cognitive tasks"""

    BASE_INPUT_DIM = 7  # Maximum input channels needed by any task
    OUTPUT_DIM = 5  # Default output dimension for all tasks
    BINARY_CHANNEL = 3  # Dedicated channel for binary (go/nogo) responses
    is_binary_task = False  # Whether accuracy is binary (BINARY_CHANNEL > 0.5)
    is_population_task = False  # Whether output is population-coded (needs decoding)
    is_argmax_task = False  # Whether accuracy uses argmax over loss_channels
    loss_channels = None  # Output channels for loss; None = all except first
    PHASE_NAMES: list = []  # Override per task with phase name strings

    def __init__(self, duration_params: Dict):
        self.duration_params = duration_params
        self.output_dim = self.__class__.OUTPUT_DIM

    def generate_trial(self):
        raise NotImplementedError

    def generate_trial_with_phases(self):
        """Generate a trial with per-timestep phase annotations.

        Returns:
            (inputs, targets, mask, phases) where phases is a np.ndarray[int]
            of shape (T_total,) with values indexing into PHASE_NAMES.
        """
        raise NotImplementedError

    def __call__(self):
        """Wrapper that automatically pads inputs to BASE_INPUT_DIM"""
        inputs, targets, mask = self.generate_trial()

        # Pad inputs if necessary
        if inputs.shape[1] < self.BASE_INPUT_DIM:
            padding = torch.zeros(
                inputs.shape[0], self.BASE_INPUT_DIM - inputs.shape[1]
            )
            inputs = torch.cat([inputs, padding], dim=1)

        return inputs, targets, mask


class HierarchicalTasksDataset(Dataset):
    """Dataset that generates trials from multiple tasks"""

    def __init__(
        self,
        tasks: List[CognitiveTask],
        n_trials: int = 1000,
        task_indices: Optional[List[int]] = None,
        fixed: bool = False,
    ):
        self.tasks = tasks
        self.n_trials = n_trials
        self.fixed = fixed

        if task_indices is None:
            self.task_indices = np.random.randint(len(tasks), size=n_trials)
        else:
            assert len(task_indices) == n_trials
            self.task_indices = task_indices

        # Pre-generate and cache all trials for fixed datasets (e.g. test sets)
        if fixed:
            self._cache = []
            for idx in range(n_trials):
                task_idx = self.task_indices[idx]
                inputs, targets, mask = self.tasks[task_idx]()
                self._cache.append((inputs, targets, mask, task_idx))

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        if self.fixed:
            return self._cache[idx]

        task_idx = self.task_indices[idx]
        task = self.tasks[task_idx]
        inputs, targets, mask = task()
        return inputs, targets, mask, task_idx


class MultiTaskDataset(Dataset):
    """Dataset that generates trials from multiple tasks"""

    def __init__(
        self,
        tasks: List[CognitiveTask],
        n_trials: int = 1000,
        task_indices: List[int] = None,
        fixed: bool = False,
    ):
        self.tasks = tasks
        self.n_trials = n_trials
        self.fixed = fixed

        if task_indices is None:
            # Random assignment if none provided
            self.task_indices = np.random.randint(len(tasks), size=n_trials)
        else:
            assert len(task_indices) == n_trials
            self.task_indices = task_indices

        # Pre-generate and cache all trials for fixed datasets (e.g. test sets)
        if fixed:
            self._cache = []
            for idx in range(n_trials):
                task_idx = self.task_indices[idx]
                inputs, targets, mask = self.tasks[task_idx]()
                task_id = torch.zeros(inputs.shape[0], len(self.tasks))
                task_id[:, task_idx] = 1
                inputs = torch.cat([inputs, task_id], dim=1)
                self._cache.append((inputs, targets, mask, task_idx))

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        if self.fixed:
            return self._cache[idx]

        task_idx = self.task_indices[idx]
        task = self.tasks[task_idx]

        # Generate trial (automatically padded via __call__)
        inputs, targets, mask = task()  # Use __call__ instead of generate_trial()

        # Add task identity to inputs (one-hot encoding)
        task_id = torch.zeros(inputs.shape[0], len(self.tasks))
        task_id[:, task_idx] = 1
        inputs = torch.cat([inputs, task_id], dim=1)

        return inputs, targets, mask, task_idx


class StratifiedBatchSampler(Sampler):
    """Sampler that ensures each batch maintains the global task proportions.

    Groups dataset indices by task, then fills each batch by drawing from
    each task proportionally.  Leftover indices that don't fill a complete
    batch are collected into a final smaller batch so no data is dropped.
    """

    def __init__(self, task_indices, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group dataset indices by task id
        self.task_to_indices: Dict[int, List[int]] = {}
        for idx, tid in enumerate(task_indices):
            tid = int(tid)
            self.task_to_indices.setdefault(tid, []).append(idx)

        total = len(task_indices)
        # Compute how many samples each task contributes per batch (fractional)
        self.task_counts: Dict[int, float] = {
            tid: len(idxs) / total * batch_size
            for tid, idxs in self.task_to_indices.items()
        }

    def __iter__(self):
        # Shuffle within each task group
        task_pools = {}
        for tid, idxs in self.task_to_indices.items():
            idxs = list(idxs)
            if self.shuffle:
                np.random.shuffle(idxs)
            task_pools[tid] = idxs

        # Track position within each task pool
        task_pos = {tid: 0 for tid in task_pools}

        # Use largest-remainder method to round fractional counts to integers
        # that sum exactly to batch_size
        floors = {tid: int(np.floor(c)) for tid, c in self.task_counts.items()}
        remainders = {tid: self.task_counts[tid] - floors[tid] for tid in floors}
        deficit = self.batch_size - sum(floors.values())
        # Give the extra slots to the tasks with the largest remainders
        sorted_by_rem = sorted(remainders, key=remainders.get, reverse=True)
        int_counts = dict(floors)
        for i in range(int(deficit)):
            int_counts[sorted_by_rem[i]] += 1

        batches = []
        exhausted = set()

        while len(exhausted) < len(task_pools):
            batch = []
            for tid in sorted(int_counts.keys()):
                if tid in exhausted:
                    continue
                pool = task_pools[tid]
                pos = task_pos[tid]
                n = int_counts[tid]
                # Take up to n indices from this task
                end = min(pos + n, len(pool))
                batch.extend(pool[pos:end])
                task_pos[tid] = end
                if end >= len(pool):
                    exhausted.add(tid)

            if batch:
                if self.shuffle:
                    np.random.shuffle(batch)
                batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.task_to_indices.values())
        return (total + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    """Custom collate function to handle variable length sequences and input dims"""
    inputs_list, targets_list, masks_list, task_ids = zip(*batch)

    # Find max length and max input dimension
    max_len = max(inp.shape[0] for inp in inputs_list)
    max_input_dim = max(inp.shape[1] for inp in inputs_list)
    batch_size = len(batch)

    # Determine max output dimension N across all targets (either (N,) or (T_i, N))
    N = max(tgt.shape[0] if tgt.dim() == 1 else tgt.shape[1] for tgt in targets_list)

    # Pad sequences
    inputs_padded = torch.zeros(batch_size, max_len, max_input_dim)
    masks_padded = torch.zeros(batch_size, max_len)
    targets_padded = torch.zeros(batch_size, max_len, N)

    for i, (inp, mask, tgt) in enumerate(zip(inputs_list, masks_list, targets_list)):
        T = inp.shape[0]
        input_dim = inp.shape[1]
        inputs_padded[i, :T, :input_dim] = inp
        masks_padded[i, :T] = mask

        # Handle both single (N,) and sequential (T, N) targets
        if tgt.dim() == 1:
            # Single target: repeat across all timesteps, pad channels if needed
            tgt_n = tgt.shape[0]
            targets_padded[i, :, :tgt_n] = tgt.unsqueeze(0).expand(max_len, -1)
        else:
            # Sequential targets: pad to max_len and max channels
            T_tgt = tgt.shape[0]
            tgt_n = tgt.shape[1]
            targets_padded[i, :T_tgt, :tgt_n] = tgt

    task_ids = torch.tensor(task_ids)

    return inputs_padded, targets_padded, masks_padded, task_ids
