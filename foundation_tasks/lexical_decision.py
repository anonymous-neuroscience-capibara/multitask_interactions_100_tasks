import itertools
from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class LexicalDecision(CognitiveTask):
    """
    Lexical decision (Meyer & Schvaneveldt, 1971) -- language / lexical access.

    A short 'string' of symbols (directions) is presented one symbol per step;
    the network reports whether the string is a 'word' (a member of a FIXED
    lexicon) or a 'nonword'. The lexicon is fixed across all trials (a
    deterministic subset of the possible strings), so the task requires learning
    a vocabulary of valid strings and recognizing membership -- a language /
    familiarity demand absent elsewhere in the battery.

    Input channels (3): [fixation, sym_cos, sym_sin].
    Output: channel 0 = fixation; channel 3 = word/nonword decision (binary).
    """

    PHASE_NAMES = ["context", "string", "response"]
    is_binary_task = True

    def __init__(self, duration_params: Dict, n_symbols: int = 4, length: int = 2):
        super().__init__(duration_params)
        self.n_symbols = n_symbols
        self.length = length
        self.loss_channels = [self.BINARY_CHANNEL]
        # Fixed lexicon: the first half of the lexicographically-sorted strings.
        all_strings = list(itertools.product(range(n_symbols), repeat=length))
        n_words = len(all_strings) // 2
        self.lexicon = [tuple(s) for s in all_strings[:n_words]]
        self.nonwords = [tuple(s) for s in all_strings[n_words:]]
        self.angles = np.linspace(0, 2 * np.pi, n_symbols, endpoint=False)

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_sym = 2
        T_response = np.random.randint(*dp["response"])
        T_str = self.length * T_sym
        T_total = T_context + T_str + T_response

        word = np.random.rand() < 0.5
        pool = self.lexicon if word else self.nonwords
        string = pool[np.random.randint(len(pool))]

        inputs = torch.zeros(T_total, 3)
        inputs[:, 0] = 1.0
        for k, sym in enumerate(string):
            t0 = T_context + k * T_sym
            inputs[t0:t0 + T_sym, 1] = np.cos(self.angles[sym])
            inputs[t0:t0 + T_sym, 2] = np.sin(self.angles[sym])
        t_resp = T_context + T_str
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[self.BINARY_CHANNEL] = 1.0 if word else 0.0

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_str, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Tr, 2, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
