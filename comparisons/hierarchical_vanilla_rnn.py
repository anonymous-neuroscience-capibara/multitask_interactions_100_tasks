"""Hierarchical VANILLA (Elman) RNN: the architectural baseline for HierarchicalPLRNN.

Same hierarchisation machinery as the joint PLRNN runs -- a per-task p_vector (P =
num_individual_params) mapped through SHARED projections to concrete per-task
parameters -- but the recurrence is the textbook vanilla RNN

    z_t = phi(W z_{t-1} + C x_t + h),   y_t = D z_t,   phi = tanh (or relu)

instead of the PLRNN's linear/ReLU split. There is no A and no L: every unit is
nonlinear through phi, so the projection set is p2W, p2h, p2C, p2D (no p2A).

model.L is set to 0 ON PURPOSE: BPTT.regularization_loss reads model.M / model.L and
with L=0 its diag_eff reduces to W_ii, i.e. exactly the MAR analogue for a full
recurrent matrix ((W_ii-1)^2 + offdiag + h^2). The scheme's get_parameters returns an
empty A of shape (B, 0) so the 5-tuple unpacking in BPTT works unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hierachical_model_task.projection_plrnn import base_hierarchisation


class vanilla_projection_hierarchisation(base_hierarchisation):
    """Hierarchisation for the vanilla RNN (full W, h, C, D; no A).

    Only p_vector (per task) and noise_cov are individual; all projection matrices
    are shared. Initialisation mirrors plrnn_projection_hierarchisation: p_vector
    uniform(-1, 1), p2W xavier-uniform at w_init_gain, the rest at gain 0.1.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params
        self.M = args.hidden_size
        self.N = args.output_size
        self.input_dim = args.obs_size

    def init_parameters(self):
        super().init_parameters()  # noise_cov, obs_matrix
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )
        self.model.p2W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.dp, self.M, self.M), gain=self.w_init_gain
            )
        )
        self.model.p2h = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dp, self.M), gain=0.1)
        )
        self.model.p2C = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.dp, self.M, self.input_dim), gain=0.1
            )
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
        Returns:
            A (batch, 0) empty, W (batch, M, M), h (batch, M),
            C (batch, M, input_dim), D (batch, N, M)
        """
        p = self.model.p_vector[subject]  # (B, dp)
        A = p.new_zeros(p.shape[0], 0)  # no A in a vanilla RNN (L = 0)
        W = torch.einsum("bd,dij->bij", p, self.model.p2W)
        h = p @ self.model.p2h
        C = torch.einsum("bd,dij->bij", p, self.model.p2C)
        D = torch.einsum("bd,dij->bij", p, self.model.p2D)
        return A, W, h, C, D

    def grouped_parameters(self):
        shared, individual = super().grouped_parameters()
        shared += [
            self.model.p2W,
            self.model.p2h,
            self.model.p2C,
            self.model.p2D,
        ]
        individual += [self.model.p_vector]
        return shared, individual


class HierarchicalVanillaRNN(nn.Module):
    """Vanilla RNN with hierarchical (p_vector-projected) per-task parameters.

    Drop-in for HierarchicalPLRNN in the BPTT trainer: same constructor signature,
    same forward(inputs, subject) contract, same hierarchisation_scheme interface.
    """

    def __init__(self, args, dataset):
        super().__init__()
        self.M = args.hidden_size
        self.L = 0  # no linear/nonlinear split; keeps BPTT.regularization_loss exact
        self.N = args.output_size
        self.input_dim = args.obs_size
        self.nonlinearity = getattr(args, "nonlinearity", "tanh")

        # Aliases for base_hierarchisation compatibility
        self.dh = self.M
        self.dz = self.M
        self.dx = self.N
        self.df = self.N

        self.hierarchisation_scheme = vanilla_projection_hierarchisation(self, args)
        self.hierarchisation_scheme.init_parameters()
        self.device = args.device
        self.to(self.device)

    def phi(self, x):
        return torch.tanh(x) if self.nonlinearity == "tanh" else F.relu(x)

    def forward(self, inputs, subject):
        """Process a batch of sequences.

        Args:
            inputs: (B, T, input_dim)
            subject: (B,) task indices
        Returns:
            outputs: (B, T, N)
        """
        B, T, _ = inputs.shape
        _, W, h, C, D = self.hierarchisation_scheme.get_parameters(subject)

        z = torch.zeros(B, self.M, device=self.device)
        z_all = torch.empty(B, T, self.M, device=self.device)

        for t in range(T):
            C_output = torch.einsum("bij,bj->bi", C, inputs[:, t])
            W_output = torch.einsum("bij,bj->bi", W, z)
            z = self.phi(W_output + C_output + h)
            z_all[:, t] = z

        outputs = torch.einsum("btj,bij->bti", z_all, D)  # (B, T, N)
        return outputs
