from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class PriorIntegration(CognitiveTask):
    """
    Prior-biased perceptual decision (Hanks, Mazurek & Shadlen 2011).

    A prior cue points to the more-likely direction; the true direction is drawn
    around that prior. Then noisy directional evidence is shown. The network
    must report the direction, integrating the (reliable) prior with the
    (noisy) evidence -- on weak-evidence trials, leaning on the prior is what
    yields accuracy. Distinct from PerceptualDM (no prior) and MultiSens
    (integration of two *sensory* cues): here it is prior x likelihood.

    Input channels (5): [fixation, prior_cos, prior_sin, ev_cos, ev_sin].
    Output (angular, channels 1,2): reported direction.
    """

    PHASE_NAMES = ["cue", "stimulus", "response"]

    def __init__(self, duration_params: Dict, prior_kappa: float = 1.2,
                 coherence_range=(0.2, 0.6)):
        super().__init__(duration_params)
        self.prior_kappa = prior_kappa  # smaller spread of true dir around prior
        self.coherence_range = coherence_range
        self.loss_channels = [1, 2]

    def _build(self):
        dp = self.duration_params
        Tc = np.random.randint(*dp["context"])
        Ts = np.random.randint(*dp["stimulus"])
        Tr = np.random.randint(*dp["response"])
        T_total = Tc + Ts + Tr

        prior_dir = np.random.uniform(0, 2 * np.pi)
        # true direction near the prior (von-Mises-like via wrapped normal)
        true_dir = prior_dir + np.random.normal(0, 1.0 / self.prior_kappa)
        coherence = np.random.uniform(*self.coherence_range)
        noise_std = 0.6 * (1 - coherence)

        inputs = torch.zeros(T_total, 5)
        inputs[:, 0] = 1.0
        # prior cue: shown during context + stimulus (persistent contextual cue)
        inputs[:Tc + Ts, 1] = np.cos(prior_dir)
        inputs[:Tc + Ts, 2] = np.sin(prior_dir)
        # noisy evidence during stimulus
        nc = np.random.normal(0, noise_std, size=Ts)
        ns = np.random.normal(0, noise_std, size=Ts)
        inputs[Tc:Tc + Ts, 3] = torch.from_numpy(
            (coherence * np.cos(true_dir) + nc).astype(np.float32))
        inputs[Tc:Tc + Ts, 4] = torch.from_numpy(
            (coherence * np.sin(true_dir) + ns).astype(np.float32))
        t_resp = Tc + Ts
        inputs[t_resp:, 0] = 0.0

        targets = torch.zeros(self.output_dim)
        targets[1] = np.cos(true_dir)
        targets[2] = np.sin(true_dir)

        mask = torch.zeros(T_total)
        mask[t_resp:] = 1.0
        return inputs, targets, mask, (Tc, Ts, Tr)

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
