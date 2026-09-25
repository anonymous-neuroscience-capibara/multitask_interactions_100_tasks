"""Hierarchical GATED RNN (LSTM / GRU): gated baselines for HierarchicalPLRNN.

Same hierarchisation machinery as the joint PLRNN/vanilla runs -- a per-task
p_vector (P = num_individual_params) mapped through SHARED projections to concrete
per-task parameters -- but the recurrence is a standard gated cell:

  LSTM:  i,f,g,o = split(W_ih x_t + W_hh h_{t-1} + b)
         c_t = sigmoid(f + 1) * c_{t-1} + sigmoid(i) * tanh(g)    (+1 = forget-bias
         h_t = sigmoid(o) * tanh(c_t)                              offset, standard)
  GRU:   r,z,n-parts = split(...)
         r = sigmoid(x_r + h_r + b_r); z = sigmoid(x_z + h_z + b_z)
         n = tanh(x_n + r * h_n + b_n);  h_t = (1 - z) * n + z * h_{t-1}

The projection set is p2Wih (dp, G*M, input_dim), p2Whh (dp, G*M, M), p2b (dp, G*M),
p2D (dp, N, M), with G = 4 (LSTM) / 3 (GRU). There is no A and no MAR analogue for
gated cells (no single recurrent diagonal to pin to 1) -- run BPTT with args.tau = 0,
which the training script enforces.

BPTT compatibility: get_parameters returns (A_empty, Whh, b, Wih, D) in the (A, W,
h, C, D) slot order BPTT unpacks. With tau=0 BPTT.regularization_loss is an exact
zero; its shape accesses (W[:, i, i] with i < M_reg <= M, h[:, i]) stay in bounds on
the stacked (S, G*M, M) / (S, G*M) tensors, so the trainer runs unchanged.

NOTE for the trajectory analyses: the LSTM's full dynamical state is (h_t, c_t) --
2M-dimensional. forward stores only h_t (what the readout sees); apples-to-apples
state-space comparisons with the M-dim PLRNN/vanilla models need that caveat.
"""

import torch
import torch.nn as nn

from hierachical_model_task.projection_plrnn import base_hierarchisation

N_GATES = {"lstm": 4, "gru": 3}
LSTM_FORGET_BIAS = 1.0


class gated_projection_hierarchisation(base_hierarchisation):
    """Hierarchisation for gated cells (stacked-gate W_ih, W_hh, b; readout D).

    Only p_vector (per task) and noise_cov are individual; all projection matrices
    are shared. Initialisation mirrors the PLRNN/vanilla schemes: p_vector
    uniform(-1, 1), p2Whh xavier-uniform at w_init_gain, the rest at gain 0.1.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params
        self.M = args.hidden_size
        self.N = args.output_size
        self.input_dim = args.obs_size
        self.G = model.G

    def init_parameters(self):
        super().init_parameters()  # noise_cov, obs_matrix
        GM = self.G * self.M
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )
        self.model.p2Wih = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.dp, GM, self.input_dim), gain=0.1
            )
        )
        self.model.p2Whh = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.dp, GM, self.M), gain=self.w_init_gain
            )
        )
        self.model.p2b = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dp, GM), gain=0.1)
        )
        self.model.p2D = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dp, self.N, self.M), gain=0.1)
        )

    def re_init(self):
        super().init_parameters()
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )

    def get_parameters(self, subject):
        """Map per-task p_vector to concrete parameters.

        Args:
            subject: LongTensor of shape (batch,) with task indices
        Returns (in BPTT's (A, W, h, C, D) slot order):
            A (batch, 0) empty, Whh (batch, G*M, M), b (batch, G*M),
            Wih (batch, G*M, input_dim), D (batch, N, M)
        """
        p = self.model.p_vector[subject]  # (B, dp)
        A = p.new_zeros(p.shape[0], 0)
        Whh = torch.einsum("bd,dij->bij", p, self.model.p2Whh)
        b = p @ self.model.p2b
        Wih = torch.einsum("bd,dij->bij", p, self.model.p2Wih)
        D = torch.einsum("bd,dij->bij", p, self.model.p2D)
        return A, Whh, b, Wih, D

    def grouped_parameters(self):
        shared, individual = super().grouped_parameters()
        shared += [
            self.model.p2Wih,
            self.model.p2Whh,
            self.model.p2b,
            self.model.p2D,
        ]
        individual += [self.model.p_vector]
        return shared, individual


class HierarchicalGatedRNN(nn.Module):
    """LSTM/GRU with hierarchical (p_vector-projected) per-task parameters.

    Drop-in for HierarchicalPLRNN in the BPTT trainer: same constructor signature,
    same forward(inputs, subject) contract, same hierarchisation_scheme interface.
    Set args.cell to 'lstm' or 'gru'; run BPTT with args.tau = 0 (no MAR analogue).
    """

    def __init__(self, args, dataset):
        super().__init__()
        self.cell_type = getattr(args, "cell", "lstm")
        if self.cell_type not in N_GATES:
            raise ValueError(f"unknown cell: {self.cell_type!r}")
        self.G = N_GATES[self.cell_type]

        self.M = args.hidden_size
        self.L = 0  # keeps BPTT.regularization_loss (tau=0) shape-safe
        self.N = args.output_size
        self.input_dim = args.obs_size

        # Aliases for base_hierarchisation compatibility
        self.dh = self.M
        self.dz = self.M
        self.dx = self.N
        self.df = self.N

        self.hierarchisation_scheme = gated_projection_hierarchisation(self, args)
        self.hierarchisation_scheme.init_parameters()
        self.device = args.device
        self.to(self.device)

    def forward(self, inputs, subject):
        """Process a batch of sequences.

        Args:
            inputs: (B, T, input_dim)
            subject: (B,) task indices
        Returns:
            outputs: (B, T, N)
        """
        B, T, _ = inputs.shape
        _, Whh, b, Wih, D = self.hierarchisation_scheme.get_parameters(subject)

        h = torch.zeros(B, self.M, device=self.device)
        h_all = torch.empty(B, T, self.M, device=self.device)
        if self.cell_type == "lstm":
            c = torch.zeros(B, self.M, device=self.device)

        # input contributions for all timesteps at once: (B, T, G*M)
        x_pre_all = torch.einsum("bij,btj->bti", Wih, inputs)

        for t in range(T):
            h_pre = torch.einsum("bij,bj->bi", Whh, h)  # (B, G*M)
            if self.cell_type == "lstm":
                pre = x_pre_all[:, t] + h_pre + b
                i, f, g, o = pre.chunk(4, dim=1)
                c = (torch.sigmoid(f + LSTM_FORGET_BIAS) * c
                     + torch.sigmoid(i) * torch.tanh(g))
                h = torch.sigmoid(o) * torch.tanh(c)
            else:  # gru: r gates only the HIDDEN contribution of the candidate n
                x_r, x_z, x_n = x_pre_all[:, t].chunk(3, dim=1)
                h_r, h_z, h_n = h_pre.chunk(3, dim=1)
                b_r, b_z, b_n = b.chunk(3, dim=1)
                r = torch.sigmoid(x_r + h_r + b_r)
                z = torch.sigmoid(x_z + h_z + b_z)
                n = torch.tanh(x_n + r * h_n + b_n)
                h = (1 - z) * n + z * h
            h_all[:, t] = h

        outputs = torch.einsum("btj,bij->bti", h_all, D)  # (B, T, N)
        return outputs
