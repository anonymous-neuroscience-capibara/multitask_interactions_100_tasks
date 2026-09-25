from typing import Dict

import numpy as np
import torch

from tasks.dataset import CognitiveTask


class FlipFlop(CognitiveTask):
    """
    N-bit flip-flop -- a memory task (Sussillo & Barak, 2013).

    Plain version: imagine N independent switches, each ON (+1) or OFF (-1).
    Now and then a switch is "flicked" (a brief +1 or -1 pulse on its input
    line). At EVERY timestep the network must report the current ON/OFF state
    of all N switches -- including the long gaps when nothing is flicked, so it
    has to *remember* the last setting of each switch.

    Example for one switch:
        input :  +1   0   0   0   -1   0   0   +1
        output:  +1  +1  +1  +1   -1  -1  -1  +1     (holds between flicks)

    Why it's a useful probe: with N switches there are 2**N possible ON/OFF
    patterns, and the network must be able to rest in any of them and stay put
    until a pulse moves it. Holding many discrete stable states like this needs
    genuine nonlinear recurrence -- a diagonal / near-linear network cannot do
    it. So it's a clean test of real memory, and the origin of the "bitcode" /
    fixed-point structure analysed elsewhere. Difficulty grows with n_bits
    (2**N states): FlipFlop2 -> 4 states, FlipFlop3 -> 8.

    Timeline (one phase, length sampled per trial):
    - t = 0: every switch gets one defining pulse (random +/-1), so all states
      are set from the very first step.
    - t > 0: each switch is independently flicked with probability `pulse_prob`;
      otherwise its input is 0 and it must hold its value.

    Channels (note the input/output offset):
    - INPUT: N channels, one pulse line per switch (0 = no flick this step,
      +1/-1 = set that switch). Padded to BASE_INPUT_DIM=7 for the model.
    - OUTPUT: channel 0 = fixation (unused here, kept 0); channels 1..N = the
      held value (+/-1) of each switch. So switch k is read IN on input channel
      k but read OUT on output channel k+1 (channel 0 is reserved for fixation).
    - LOSS / ACCURACY: use loss_channels = [1..N] (the bit channels). Loss is
      the shared masked MSE toward the +/-1 held values; accuracy = fraction of
      timesteps where the sign of EVERY switch is correct (the is_flipflop_task
      branch in bptt.py / rnn_model.py). All timesteps are scored.
    """

    PHASE_NAMES = ["run"]
    is_flipflop_task = True  # per-timestep sign match on loss_channels

    def __init__(
        self,
        duration_params: Dict,
        n_bits: int = 3,
        seq_len_range=(30, 50),
        pulse_prob: float = 0.1,
    ):
        super().__init__(duration_params)
        self.n_bits = n_bits
        self.seq_len_range = seq_len_range
        self.pulse_prob = pulse_prob
        if n_bits > self.OUTPUT_DIM - 1:
            raise ValueError(
                f"n_bits={n_bits} needs {n_bits + 1} output channels but "
                f"OUTPUT_DIM={self.OUTPUT_DIM}. Reduce n_bits or raise OUTPUT_DIM."
            )
        # Bits live in output channels 1..n_bits (channel 0 is fixation).
        self.loss_channels = list(range(1, n_bits + 1))

    def _sample_trial(self):
        T = int(np.random.randint(*self.seq_len_range))
        N = self.n_bits

        # Pulse train: (T, N), values in {-1, 0, +1}.
        pulses = np.zeros((T, N), dtype=np.float32)
        pulses[0] = np.random.choice([-1.0, 1.0], size=N)  # defining pulse
        fire = np.random.rand(T, N) < self.pulse_prob
        signs = np.random.choice([-1.0, 1.0], size=(T, N))
        pulses[1:][fire[1:]] = signs[1:][fire[1:]]

        # Held state: carry the last non-zero pulse forward.
        state = np.zeros((T, N), dtype=np.float32)
        current = pulses[0].copy()
        for t in range(T):
            nz = pulses[t] != 0
            current[nz] = pulses[t][nz]
            state[t] = current

        return T, N, pulses, state

    def generate_trial(self):
        T, N, pulses, state = self._sample_trial()

        inputs = torch.zeros(T, N)
        inputs[:, :N] = torch.from_numpy(pulses)

        targets = torch.zeros(T, self.output_dim)  # channel 0 (fixation) stays 0
        targets[:, 1 : N + 1] = torch.from_numpy(state)

        mask = torch.ones(T)
        return inputs, targets, mask

    def generate_trial_with_phases(self):
        inputs, targets, mask = self.generate_trial()
        phases = np.zeros(inputs.shape[0], dtype=np.int64)  # single "run" phase
        return inputs, targets, mask, phases
