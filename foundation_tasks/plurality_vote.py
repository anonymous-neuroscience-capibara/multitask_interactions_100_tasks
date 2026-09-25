from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class PluralityVote(CognitiveTask):
    """
    Plurality vote / statistical mode over a categorical stream.

    A stream of L discrete symbols (each one of K categories) is presented one
    at a time on K one-hot channels; after a delay the network reports the MOST
    FREQUENT symbol (the mode). This requires maintaining K parallel running
    counts and a winner-take-all comparison at report time -- a K-way
    competitive accumulation distinct from CountBin (bins a single scalar
    count), MaxOfK (parallel *simultaneous* evidence, not a temporal tally) and
    StimulusIdentity (memory for a single item, no counting).

    Input channels (1 + K): [fixation, sym_0, ..., sym_{K-1}].
    Output (argmax over the last K channels): the winning category.
    Accuracy: argmax over loss_channels at the last masked step (chance = 1/K).
    """

    PHASE_NAMES = ["context", "stream", "delay", "response"]

    def __init__(self, duration_params: Dict, n_classes: int = 3,
                 stream_len: int = 6):
        super().__init__(duration_params)
        self.is_argmax_task = True
        self.n_classes = int(n_classes)
        self.stream_len = int(stream_len)
        self.loss_channels = list(range(-self.n_classes, 0))

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        T_item = 2
        Td = np.random.randint(*dp["delay"])
        Tr = np.random.randint(*dp["response"])
        K, L = self.n_classes, self.stream_len
        T_stream = L * T_item
        T_total = Tc + T_stream + Td + Tr

        # Pick the winner UNIFORMLY first (so the mode class is balanced, not
        # biased by argmax tie-breaking), then force it to be the strict mode.
        w = int(np.random.randint(K))
        syms = np.random.randint(0, K, size=L)
        while True:
            counts = np.bincount(syms, minlength=K)
            others = counts.copy()
            others[w] = -1
            if counts[w] > others.max():
                break
            cand = np.where(syms != w)[0]  # promote one non-winner item to w
            syms[cand[0]] = w
        np.random.shuffle(syms)  # counts (hence winner) are order-invariant
        cls = w

        inputs = torch.zeros(T_total, 1 + K)
        inputs[:Tc + T_stream + Td, 0] = 1.0
        for k in range(L):
            t0 = Tc + k * T_item
            inputs[t0:t0 + T_item, 1 + int(syms[k])] = 1.0
        t_resp = Tc + T_stream + Td

        targets = torch.zeros(self.output_dim)
        targets[self.loss_channels[cls]] = 1.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (Tc, T_stream, Td, Tr)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Td, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Td, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
