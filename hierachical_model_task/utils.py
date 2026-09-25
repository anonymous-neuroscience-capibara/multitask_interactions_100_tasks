from typing import Optional
from dataclasses import dataclass, field


@dataclass
class HiearchicalModelConfig:
    tasks: list = field(default_factory=list)
    num_subjects: Optional[int] = (
        None  # Number of individual-specific parameters (defaults to len(tasks))
    )
    num_individual_params: int = 6  # Dimension of individual parameter vector
    obs_size: int = 7  # Input observation size (BASE_INPUT_DIM)
    output_size: int = 5  # Output size (OUTPUT_DIM)
    hidden_size: int = 32  # Hidden layer size (M)
    nonlinear_units: int = 2  # Nonlinear layer size (L)
    m_reg_override: Optional[int] = None  # if set, overrides M_reg (default = hidden_size // 2)
    batch_size: int = 64  # Batch size for the train/test DataLoaders
    tau: float = 0.01  # Time constant for the model
    num_epochs: int = 300  # Number of training epochs
    learning_rate: float = 0.001  # Learning rate for shared parameters
    individual_learning_rate: Optional[float] = (
        0.001  # Learning rate for individual parameters
    )
    weight_decay: float = 0.0  # Weight decay for optimizer
    grad_clip: float = 0.0  # Max global grad-norm for clipping (0 = off)
    a_clamp: float = 0.0  # Upper bound for the latent self-recurrence A (0 = off).
    # When > 0, A is clamped to [0, a_clamp] every forward pass, so the linear
    # self-loop z_j <- A_j*z_j can't diverge. Use ~0.999 to keep near-integrator
    # dynamics while guaranteeing contraction.
    lr_warmup_epochs: int = 0  # Linearly ramp LR from 0 -> full over this many
    # epochs (0 = off). Prevents early-training explosions in big models.
    w_init_gain: float = 0.1  # xavier gain for the recurrent projection p2W.
    # Smaller -> more contractive W at init (helps big-M stability). 0.1 = original.
    use_gpu: bool = False  # Flag to use GPU
    device_id: int = 0  # GPU device ID
    compile: bool = False  # Flag to compile the model
    # data_path: str = "data/training_data.pt"  # Path to training data
    # trial_to_subject_path: str = "data/task_mapping.pt"  # Path to trial-to-subject mapping
    # num_onehot_subjects: int = 11  # Number of one-hot encoded subjects
    obs_model: str = "identity"  # Observation model type
    learn_noise_cov: bool = False  # Flag to learn noise covariance
    tf_alpha_start: float = 0.0  # Teacher forcing alpha start
    tf_alpha_end: float = 0.0  # Teacher forcing alpha end
    compile: bool = False  # Flag to compile the model
    device: str = "cpu"  # Device to run the model on
    early_stopping_patience: int = (
        50  # Stop training if no improvement for this many epochs
    )
    early_stopping_start: int = (
        0  # Warmup: train at least this many epochs before early stopping can fire
    )
    hierarchisation: str = "all"  # Which params to hierarchise: all, AW, CD

    def __post_init__(self):
        """Set num_subjects from len(tasks) if not explicitly provided."""
        if self.num_subjects is None and self.tasks:
            self.num_subjects = len(self.tasks)

    @property
    def M_reg(self):
        """Regularization dimension. Defaults to half the hidden units
        (hidden_size // 2), evaluated on access so it always reflects the
        current M -- even when hidden_size is set after construction. Can be
        overridden by assigning `.M_reg = <int>` (stored in m_reg_override)."""
        if self.m_reg_override is not None:
            return self.m_reg_override
        return self.hidden_size // 2

    @M_reg.setter
    def M_reg(self, value):
        self.m_reg_override = value

    @property
    def full_learning_rate(self):
        return (
            self.learning_rate,
            (
                self.individual_learning_rate
                if self.individual_learning_rate is not None
                else self.learning_rate
            ),
        )
