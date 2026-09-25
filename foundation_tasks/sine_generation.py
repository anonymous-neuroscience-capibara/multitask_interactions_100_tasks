from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class SineGeneration(CognitiveTask):
    """
    Frequency-cued oscillator / sine generation -- a limit-cycle task
    (Sussillo & Abbott, 2009).

    Plain version: the input names a frequency (a single number); the network
    must then produce a steady oscillation at that frequency and keep it going.
    The output is a point travelling around the unit circle, (cos(w*t),
    sin(w*t)) -- a fast w spins quickly, a slow w crawls. A correct solution is
    a *limit cycle* (a stable periodic orbit) in latent space: the rotation /
    "loop" counterpart to the flip-flop's discrete fixed points / "stars".

    Why it probes nonlinearity: a linear system can oscillate at ONE fixed
    frequency, but (1) producing a *cued, tunable* frequency from a constant
    input and (2) a *self-correcting, amplitude-stable* oscillation both need a
    genuine nonlinear limit cycle -- a near-linear rotation is marginally stable
    and single-frequency.

    Timeline:
    - Context: fixation on; frequency cue on; not scored.
    - Response: fixation off (go); frequency cue still on; output oscillates and
      is scored.

    Channels:
    - INPUT (2): channel 0 = fixation; channel 1 = freq_level in [0, 1] (the
      cued frequency, held the whole trial). Padded to BASE_INPUT_DIM=7.
    - OUTPUT: channel 0 = fixation; channels 1, 2 = (cos, sin) of the phase.
    - NOTE: input channel 1 (freq) and output channel 1 (cos) are DIFFERENT
      things -- inputs and outputs are separate tensors that share only the
      index, never data. cos/sin live on output channels 1,2 to match the
      battery-wide angular convention (accuracy reads atan2(ch2, ch1)).
    - LOSS / ACCURACY: loss_channels = [1, 2]. Loss = shared masked MSE;
      accuracy = per-timestep angular match (is_copytask branch): fraction of
      response steps whose phase is within threshold of the target.
    """

    PHASE_NAMES = ["context", "response"]
    # Reuse the per-timestep angular accuracy path (atan2 on channels 1,2).
    is_copytask = True

    def __init__(
        self,
        duration_params: Dict,
        n_freqs: int = 4,
        omega_range=(0.15, 0.6),  # radians per timestep
    ):
        super().__init__(duration_params)
        self.omegas = np.linspace(omega_range[0], omega_range[1], n_freqs)
        self.omega_max = float(omega_range[1])
        self.loss_channels = [1, 2]

    def _build(self):
        T_context = np.random.randint(*self.duration_params["context"])
        T_response = np.random.randint(*self.duration_params["response"])
        # Lengthen the response so several cycles are produced.
        T_response = max(T_response, 20)
        T_total = T_context + T_response

        omega = float(np.random.choice(self.omegas))
        freq_level = omega / self.omega_max

        inputs = torch.zeros(T_total, 2)
        inputs[:T_context, 0] = 1.0  # fixation during context
        inputs[:, 1] = freq_level  # frequency cue throughout

        targets = torch.zeros(T_total, self.output_dim)
        targets[:T_context, 0] = 1.0  # hold fixation during context

        t = np.arange(T_response)
        phase = omega * t
        targets[T_context:, 1] = torch.from_numpy(np.cos(phase).astype(np.float32))
        targets[T_context:, 2] = torch.from_numpy(np.sin(phase).astype(np.float32))

        mask = torch.zeros(T_total)
        mask[T_context:] = 1.0
        return inputs, targets, mask, T_context, T_response

    def generate_trial(self):
        inputs, targets, mask, _, _ = self._build()
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask, T_context, T_response = self._build()
        phases = np.concatenate(
            [
                np.full(T_context, 0, dtype=np.int64),
                np.full(T_response, 1, dtype=np.int64),
            ]
        )
        return inputs, targets, mask, phases
