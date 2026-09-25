import torch
import torch.nn as nn
import torch.nn.functional as F
from hierachical_model_task.projection_plrnn import (
    plrnn_projection_hierarchisation,
    plrnn_projection_AW_only,
    plrnn_projection_CD_only,
)


class HierarchicalPLRNN(nn.Module):
    """PLRNN variant (A diagonal on latent part, full W, h, C, D) with hierarchical params."""

    def __init__(self, args, dataset):
        super().__init__()
        self.M = args.hidden_size
        self.L = args.nonlinear_units if args.nonlinear_units is not None else 0
        self.N = args.output_size
        self.input_dim = args.obs_size

        # Aliases for base_hierarchisation compatibility
        self.dh = self.M  # hidden size
        self.dz = self.M  # latent state size (same as hidden for PLRNN)
        self.dx = self.N  # observation size
        self.df = self.N  # forcing size (same as observation)

        # helpers (plotter/evaluator/saver) to stay compatible with existing tooling
        # self.plotter = Plotter(self, dataset)
        # self.evaluator = Evaluator(self, args, dataset)
        # self.saver = Saver(self, args, dataset)
        # hierarchisation scheme selection
        hier_mode = getattr(args, "hierarchisation", "all")
        if hier_mode == "AW":
            self.hierarchisation_scheme = plrnn_projection_AW_only(self, args)
        elif hier_mode == "CD":
            self.hierarchisation_scheme = plrnn_projection_CD_only(self, args)
        else:
            self.hierarchisation_scheme = plrnn_projection_hierarchisation(self, args)
        self.hierarchisation_scheme.init_parameters()
        self.device = args.device
        self.to(self.device)

    def forward(self, inputs, subject):
        """Process a batch of sequences.

        Args:
            inputs: (B, T, input_dim)
            subject: (B,) subject indices (already mapped trial->subject)
        Returns:
            outputs: (B, T, N)
        """
        B, T, _ = inputs.shape
        A, W, h, C, D = self.hierarchisation_scheme.get_parameters(subject)

        z = torch.zeros(B, self.M, device=self.device)
        z_all = torch.empty(B, T, self.M, device=self.device)

        for t in range(T):
            if self.L > 0:
                z_non_latent = z[:, : -self.L]
                z_latent = z[:, -self.L :]
                z_latent_scaled = A * z_latent
                z_latent_act = F.relu(z_latent)
                z_combined = torch.cat([z_non_latent, z_latent_act], dim=1)
                z_update_latent = z_latent_scaled
                z_update = torch.cat(
                    [torch.zeros_like(z_non_latent), z_update_latent], dim=1
                )
            else:
                z_combined = z
                z_update = torch.zeros_like(z)

            # Batched matrix multiplications for per-subject parameters
            # C = (B, M, input_dim)
            # inputs[:, t] = (B, input_dim)
            C_output = torch.einsum("bij,bj->bi", C, inputs[:, t])
            W_output = torch.einsum("bij,bj->bi", W, z_combined)
            # W = (B, M, M)
            z = z_update + W_output + C_output + h
            z_all[:, t] = z

        # Project all hidden states through D at once
        outputs = torch.einsum("btj,bij->bti", z_all, D)  # (B, T, N)
        return outputs
