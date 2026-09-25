import torch
from torch import nn
import matplotlib.pyplot as plt


class base_hierarchisation:
    def __init__(self, model, args):
        """Base class for hierarchisation schemes."""
        self.model = model
        self.dh = model.dh
        self.dx = model.dx
        self.dz = model.dz
        self.df = model.df
        self.obs_model = args.obs_model
        self.learn_noise_cov = args.learn_noise_cov
        self.num_subjects = args.num_subjects
        # Optional hard bound on the latent self-recurrence A (0 = off). When > 0,
        # get_parameters clamps A to [0, a_clamp] so the linear self-loop can't diverge.
        self.a_clamp = getattr(args, "a_clamp", 0.0)
        # xavier gain for the recurrent projection p2W (smaller = more contractive
        # W at init, helps big-M stability). Defaults to the original 0.1.
        self.w_init_gain = getattr(args, "w_init_gain", 0.1)
        assert self.obs_model in ["identity", "linear"]

    def _clamp_A(self, A):
        """Bound A to [0, a_clamp] if enabled; otherwise return A unchanged."""
        if self.a_clamp and self.a_clamp > 0:
            return A.clamp(min=0.0, max=self.a_clamp)
        return A

    def re_init(self):
        """Re-initializes all subject specific parameters for finetuning."""
        self.model.noise_cov = nn.Parameter(
            torch.zeros(size=(self.num_subjects, 1, self.dx))
        )

    def init_parameters(self):
        """Initializes the model parameters for the given hierarchisation scheme.
        Initializes the covariance of the gaussian noise model to all ones
        (Here: log cov -> to zeros)."""
        self.model.noise_cov = nn.Parameter(
            torch.zeros(size=(self.num_subjects, 1, self.dx)),
            requires_grad=self.learn_noise_cov,
        )
        # obs model
        if self.obs_model == "identity":
            self.model.obs_matrix = nn.Parameter(
                torch.eye(self.dz, self.dx), requires_grad=False
            )
        else:
            self.model.obs_matrix = nn.Parameter(torch.randn(self.dz, self.dx))

    def get_parameters(self, subject):
        """Constructs and returns the model parameters.
        Args:
            subject: subject index for each sample in the batch
        returns:
            model parameters for the given subject(s)"""
        pass

    def grouped_parameters(self):
        """Returns a generator object for the individual and the shared
        parameters respectively."""
        return [self.model.obs_matrix], [self.model.noise_cov]

    @torch.no_grad()
    def plot_stuff(self):
        """Function for viualizing interesting things. This is called by a plotter object
        to write it to tensorboard. Plots the noise covariance as an image"""
        fig, ax = plt.subplots(1, figsize=(self.num_subjects + 1, self.dz))
        c = ax.imshow(self.model.noise_cov.squeeze(1).cpu().T)
        fig.colorbar(c, ax=ax)
        ax.set_ylabel(r"$\sigma^2_i$")
        ax.set_xlabel("subject")
        return [(fig, "noise_covariance")]

    def loss(self):
        """Defines any regularization losses on the parameters."""
        return 0.0


class plrnn_projection_hierarchisation(base_hierarchisation):
    """Hierarchisation scheme for the PLRNN variant (A diag, full W, h, C, D).

    Shared projections (p2*) map a per-subject vector p into the actual
    parameter tensors. Only p_vector (per subject) and noise_cov are individual;
    all projection matrices are shared.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params  # dimension of individual parameter vector
        self.M = args.hidden_size
        self.L = args.nonlinear_units if args.nonlinear_units is not None else 0
        self.N = args.output_size
        self.input_dim = args.obs_size

    def init_parameters(self):
        """Initialize shared projections and per-subject vectors."""
        super().init_parameters()  # noise_cov
        # individual p_vector per subject
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )
        # here, we add a third dimension (dp) because I need to multiply it a vector an still having a matrix
        self.model.p2A = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dp, self.L), gain=0.1)
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
        """Map per-subject p_vector to concrete parameters.

        Args:
            subject: LongTensor of shape (batch,) with subject indices
        Returns:
            A (batch, L), W (batch, M, M), h (batch, M), C (batch, M, input_dim), D (batch, N, M)
        """
        p = self.model.p_vector[subject]  # (B, dp)
        A = self._clamp_A(p @ self.model.p2A)  # (B, dp) @ (dp, L) -> (B, L)
        W = torch.einsum(
            "bd,dij->bij", p, self.model.p2W
        )  # (B, dp) @ (M, M, dp) -> (B, M, M)
        h = p @ self.model.p2h  # (B, dp) @ (dp, M) -> (B, M)
        C = torch.einsum(
            "bd,dij->bij", p, self.model.p2C
        )  # (B, dp) @ (M, input_dim, dp) -> (B, M, input_dim)
        D = torch.einsum(
            "bd,dij->bij", p, self.model.p2D
        )  # (B, dp) @ (N, M, dp) -> (B, N, M)
        return A, W, h, C, D

    def grouped_parameters(self):
        shared, individual = super().grouped_parameters()
        shared += [
            self.model.p2A,
            self.model.p2W,
            self.model.p2h,
            self.model.p2C,
            self.model.p2D,
        ]
        individual += [self.model.p_vector]
        return shared, individual


class plrnn_projection_AW_only(base_hierarchisation):
    """Hierarchise only the dynamics (A, W) via per-task p_vector.

    h, C, and D are shared across all tasks (plain learnable parameters).
    This is Ablation 2b: tests whether inter-task variability lives
    in the recurrent dynamics.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params
        self.M = args.hidden_size
        self.L = args.nonlinear_units if args.nonlinear_units is not None else 0
        self.N = args.output_size
        self.input_dim = args.obs_size

    def init_parameters(self):
        super().init_parameters()
        # per-task feature vector (drives only A, W, h)
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )
        # projections for dynamics only (A, W hierarchised; h shared)
        self.model.p2A = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.dp, self.L), gain=0.1)
        )
        self.model.p2W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.dp, self.M, self.M), gain=self.w_init_gain
            )
        )
        # shared h, C, D (not hierarchised)
        self.model.h_shared = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.M), gain=0.1).squeeze(0)
        )
        self.model.C_shared = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.M, self.input_dim), gain=0.1)
        )
        self.model.D_shared = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.N, self.M), gain=0.1)
        )

    def re_init(self):
        super().init_parameters()
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )

    def get_parameters(self, subject):
        B = subject.shape[0]
        p = self.model.p_vector[subject]
        A = self._clamp_A(p @ self.model.p2A)
        W = torch.einsum("bd,dij->bij", p, self.model.p2W)
        h = self.model.h_shared.unsqueeze(0).expand(B, -1)
        C = self.model.C_shared.unsqueeze(0).expand(B, -1, -1)
        D = self.model.D_shared.unsqueeze(0).expand(B, -1, -1)
        return A, W, h, C, D

    def grouped_parameters(self):
        shared, individual = super().grouped_parameters()
        shared += [
            self.model.p2A,
            self.model.p2W,
            self.model.h_shared,
            self.model.C_shared,
            self.model.D_shared,
        ]
        individual += [self.model.p_vector]
        return shared, individual


class plrnn_projection_CD_only(base_hierarchisation):
    """Hierarchise only the input/output mapping (C, D) via per-task p_vector.

    A, W, and h are shared across all tasks (plain learnable parameters).
    This is Ablation 2a: tests whether inter-task variability lives
    in the stimulus encoding and response readout.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params
        self.M = args.hidden_size
        self.L = args.nonlinear_units if args.nonlinear_units is not None else 0
        self.N = args.output_size
        self.input_dim = args.obs_size

    def init_parameters(self):
        super().init_parameters()
        # per-task feature vector (drives only C, D)
        self.model.p_vector = nn.Parameter(
            torch.repeat_interleave(
                torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0
            )
        )
        # shared dynamics (not hierarchised)
        self.model.A_shared = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.L), gain=0.1).squeeze(0)
        )
        self.model.W_shared = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.M, self.M), gain=self.w_init_gain)
        )
        self.model.h_shared = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.M), gain=0.1).squeeze(0)
        )
        # projections for I/O only
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
        B = subject.shape[0]
        p = self.model.p_vector[subject]
        A = self._clamp_A(self.model.A_shared.unsqueeze(0).expand(B, -1))
        W = self.model.W_shared.unsqueeze(0).expand(B, -1, -1)
        h = self.model.h_shared.unsqueeze(0).expand(B, -1)
        C = torch.einsum("bd,dij->bij", p, self.model.p2C)
        D = torch.einsum("bd,dij->bij", p, self.model.p2D)
        return A, W, h, C, D

    def grouped_parameters(self):
        shared, individual = super().grouped_parameters()
        shared += [
            self.model.A_shared,
            self.model.W_shared,
            self.model.h_shared,
            self.model.p2C,
            self.model.p2D,
        ]
        individual += [self.model.p_vector]
        return shared, individual
