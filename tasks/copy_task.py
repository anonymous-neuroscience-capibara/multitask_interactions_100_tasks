from typing import Dict

import numpy as np
import torch

from .dataset import CognitiveTask


class CopyTask(CognitiveTask):
    is_copytask = True
    PHASE_NAMES = ["context", "encoding", "delay", "recall"]
    """
    Copy Task: Memorize a sequence of symbols and reproduce immediately

    Timeline:
    - Encoding: present sequence of symbols (one per timestep) with fixation ON
    - Response: reproduce sequence (fixation OFF = signal to output)

    Symbol encoding strategy:
    Instead of one-hot discrete symbols, we encode each symbol as a unique
    continuous value or direction in a circular space to match the continuous
    task framework.

    For num_symbols=4:
    - Symbol 0 (blank): magnitude=0
    - Symbol 1: θ=0°    → (cos=1.0, sin=0.0)
    - Symbol 2: θ=90°   → (cos=0.0, sin=1.0)
    - Symbol 3: θ=180°  → (cos=-1.0, sin=0.0)
    - Symbol 4: θ=270°  → (cos=0.0, sin=-1.0)

    Input channels (3): [fixation, symbol_cos, symbol_sin]
    Output: [fixation, symbol_cos, symbol_sin] per timestep

    Tests: Sequential memory + temporal order
    """

    def __init__(
        self,
        duration_params: Dict,
        seq_len: int = 3,
        num_symbols: int = 4,
        delay_range: int = 0,
    ):
        super().__init__(duration_params)
        self.seq_len = seq_len
        self.num_symbols = num_symbols
        self.delay_range = delay_range
        self.loss_channels = [1, 2]

        # Pre-compute symbol angles (evenly spaced around circle)
        # Skip symbol 0 (reserved for blank/cue)
        self.symbol_angles = {
            i: 2 * np.pi * (i - 1) / num_symbols for i in range(1, num_symbols + 1)
        }
        self.symbol_angles[0] = None  # blank/padding

    def generate_trial(self):
        # Sample durations
        T_context = np.random.randint(*self.duration_params["context"])
        T_item = 1  # Time per symbol during encoding
        T_encoding = self.seq_len * T_item
        T_item_response = 1  # Time per symbol during response
        T_delay = self.delay_range
        T_response = self.seq_len
        T_total = T_context + T_encoding + T_delay + T_response

        # Sample a random sequence of symbols (excluding 0)
        while True:
            sequence = np.random.randint(1, self.num_symbols + 1, size=self.seq_len)
            if len(np.unique(sequence)) >= max(2, self.seq_len // 2):
                break

        # Create inputs: [fixation, symbol_cos, symbol_sin]
        inputs = torch.zeros(T_total, 3)

        # ===================================================================
        # PHASE 0: CONTEXT - Fixation only
        # ===================================================================
        inputs[:T_context, 0] = 1  # fixation

        # ===================================================================
        # PHASE 1: ENCODING - Present sequence of symbols (fixation ON)
        # ===================================================================
        for i, symbol in enumerate(sequence):
            t_item_start = T_context + i * T_item
            t_item_end = t_item_start + T_item

            theta = self.symbol_angles[symbol]

            inputs[t_item_start:t_item_end, 0] = 1  # fixation
            inputs[t_item_start:t_item_end, 1] = np.cos(theta)  # symbol_cos
            inputs[t_item_start:t_item_end, 2] = np.sin(theta)  # symbol_sin

        # ===================================================================
        # PHASE 2: RESPONSE - Reproduce sequence (fixation OFF)
        # ===================================================================
        t_resp_start = T_context + T_encoding + T_delay

        # Create targets with continuous angles for each symbol
        # Shape: [T_total, 3] where each row is [fixation, cos, sin]
        targets_full = torch.zeros(T_total, self.output_dim)

        for i, symbol in enumerate(sequence):
            t_item_start = t_resp_start + i * T_item_response
            t_item_end = t_item_start + T_item_response

            theta = self.symbol_angles[symbol]

            # During each response window, output the corresponding symbol as angle
            targets_full[t_item_start:t_item_end, 0] = 0  # no fixation
            targets_full[t_item_start:t_item_end, 1] = np.cos(theta)
            targets_full[t_item_start:t_item_end, 2] = np.sin(theta)

        # Mask: only evaluate during response period
        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        return inputs, targets_full, mask

    def generate_trial_with_phases(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_item = 1
        T_encoding = self.seq_len * T_item
        T_item_response = 1
        T_delay = self.delay_range
        T_response = self.seq_len
        T_total = T_context + T_encoding + T_delay + T_response

        while True:
            sequence = np.random.randint(1, self.num_symbols + 1, size=self.seq_len)
            if len(np.unique(sequence)) >= max(2, self.seq_len // 2):
                break

        inputs = torch.zeros(T_total, 3)

        inputs[:T_context, 0] = 1

        for i, symbol in enumerate(sequence):
            t_item_start = T_context + i * T_item
            t_item_end = t_item_start + T_item

            theta = self.symbol_angles[symbol]

            inputs[t_item_start:t_item_end, 0] = 1
            inputs[t_item_start:t_item_end, 1] = np.cos(theta)
            inputs[t_item_start:t_item_end, 2] = np.sin(theta)

        t_resp_start = T_context + T_encoding + T_delay

        targets_full = torch.zeros(T_total, self.output_dim)

        for i, symbol in enumerate(sequence):
            t_item_start = t_resp_start + i * T_item_response
            t_item_end = t_item_start + T_item_response

            theta = self.symbol_angles[symbol]

            targets_full[t_item_start:t_item_end, 0] = 0
            targets_full[t_item_start:t_item_end, 1] = np.cos(theta)
            targets_full[t_item_start:t_item_end, 2] = np.sin(theta)

        mask = torch.zeros(T_total)
        mask[t_resp_start:] = 1

        phase_parts = [
            np.full(T_context, 0, dtype=np.int64),
            np.full(T_encoding, 1, dtype=np.int64),
        ]
        if T_delay > 0:
            phase_parts.append(np.full(T_delay, 2, dtype=np.int64))
        phase_parts.append(np.full(T_response, 3, dtype=np.int64))
        phases = np.concatenate(phase_parts)

        return inputs, targets_full, mask, phases
