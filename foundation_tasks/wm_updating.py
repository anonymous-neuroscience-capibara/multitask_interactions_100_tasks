from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class WMUpdating(CognitiveTask):
    """
    Working-memory updating / keep-track (Yntema 1963; Miyake et al. 2000
    'updating' executive function).

    A stream of items is presented, each tagged with one of two categories.
    At the end a query names a category, and the network must report the
    DIRECTION of the MOST RECENT item of that category. This requires
    continuously updating (gating) a separate running memory per category and
    reading out the latest one -- distinct from n-back (compare to n-back) and
    Sternberg (a static held set): here memory must be overwritten as new
    same-category items arrive.

    Input channels (5): [fixation, item_cos, item_sin, category, query].
    Output: channels 1, 2 = (cos, sin) of the queried category's latest item.
    """

    PHASE_NAMES = ["context", "stream", "query", "response"]

    def __init__(self, duration_params: Dict, stream_len: int = 6, n_angles: int = 8):
        super().__init__(duration_params)
        self.stream_len = stream_len
        self.n_angles = n_angles
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        T_context = np.random.randint(*dp["context"])
        T_item = 2
        T_query = 2
        T_response = np.random.randint(*dp["response"])
        T_stream = self.stream_len * T_item
        T_total = T_context + T_stream + T_query + T_response

        angles = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        dirs = np.random.randint(self.n_angles, size=self.stream_len)
        cats = np.random.choice([-1.0, 1.0], size=self.stream_len)
        # Guarantee both categories appear so either query is answerable.
        cats[0], cats[1] = -1.0, 1.0
        query_cat = float(np.random.choice([-1.0, 1.0]))
        last_idx = max(k for k in range(self.stream_len) if cats[k] == query_cat)
        ans = dirs[last_idx]

        inputs = torch.zeros(T_total, 5)
        inputs[:, 0] = 1.0
        for k in range(self.stream_len):
            t0 = T_context + k * T_item
            inputs[t0:t0 + T_item, 1] = np.cos(angles[dirs[k]])
            inputs[t0:t0 + T_item, 2] = np.sin(angles[dirs[k]])
            inputs[t0:t0 + T_item, 3] = cats[k]          # category tag
        t_q = T_context + T_stream
        inputs[t_q:t_q + T_query, 4] = query_cat          # query cue
        t_resp = t_q + T_query
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(angles[ans])
        targets[2] = np.sin(angles[ans])

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (T_context, T_stream, T_query, T_response)

    def generate_trial(self):
        inputs, targets, mask, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, (Tc, Ts, Tq, Tr) = self._build()
        phases = np.concatenate([
            np.full(Tc, 0, dtype=np.int64), np.full(Ts, 1, dtype=np.int64),
            np.full(Tq, 2, dtype=np.int64), np.full(Tr, 3, dtype=np.int64),
        ])
        return inputs, targets, mask, phases
