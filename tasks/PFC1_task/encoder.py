import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedConvolutions(nn.Module):
    def __init__(self, dim_x, dim_z, sample_rec=False, normalize_output=True, 
                 smoothness_weight=.2):
        """
        Encoder using stacked convolutions to map from spike counts to latent states.
        Includes temporal smoothing and explicit smoothness constraints.
        Maintains the same temporal resolution as the input.
        
        Args:
            dim_x: Input dimension (number of neurons)
            dim_z: Output dimension (latent dimension)
            sample_rec: Whether to sample from recognition distribution
            normalize_output: Whether to normalize the output using layer normalization
            smoothness_weight: Weight for the temporal smoothness loss
        """
        super(StackedConvolutions, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.sample_rec = sample_rec
        self.smoothness_weight = smoothness_weight
        
        # Initial temporal smoothing layer with large kernel
        # For kernel_size=15, padding=7 maintains sequence length
        self.smooth = nn.Conv1d(dim_x, dim_x, kernel_size=15, padding=7, groups=dim_x)
        
        # Convolutional layers with increasing receptive fields
        # Calculate padding for each layer to maintain sequence length
        # For kernel_size=7:
        # - dilation=1: padding=3
        # - dilation=2: padding=6
        # - dilation=4: padding=12
        self.conv1 = nn.Conv1d(dim_x, 32, kernel_size=7, padding=3, dilation=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=6, dilation=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=12, dilation=4)
        
        # Final projection to latent space using 1x1 convolution
        self.proj = nn.Conv1d(128, dim_z, kernel_size=1, padding=0)  # 1x1 convolution maintains sequence length
        
        # Layer normalization for output
        self.normalize_output = normalize_output
        if normalize_output:
            self.layer_norm = nn.LayerNorm(dim_z)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if isinstance(module, nn.Conv1d) and module.groups == module.in_channels:
                # Initialize smoothing layer with Gaussian-like weights
                kernel_size = module.kernel_size[0]
                sigma = kernel_size / 6.0
                x = torch.arange(-(kernel_size//2), kernel_size//2 + 1)
                weights = torch.exp(-(x**2) / (2*sigma**2))
                weights = weights / weights.sum()
                module.weight.data = weights.view(1, 1, -1).repeat(module.in_channels, 1, 1)
            else:
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def compute_smoothness_loss(self, z):
        """Compute temporal smoothness loss using finite differences"""
        # Compute first-order differences
        diff = z[:, 1:] - z[:, :-1]
        # L2 loss on differences
        smoothness_loss = torch.mean(diff**2)
        return smoothness_loss
    
    def forward(self, x):
        # Input shape: (batch_size, T, dim_x)
        # Reshape for convolutions: (batch_size, dim_x, T)
        x = x.transpose(1, 2)
        
        # Apply initial temporal smoothing
        x = F.relu(self.smooth(x))
        
        # Apply convolutions with ReLU activations
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        # Project to latent space using 1x1 convolution
        z = self.proj(h)  # (batch_size, dim_z, T)
        
        # Transpose back to (batch_size, T, dim_z)
        z = z.transpose(1, 2)
        
        # Apply layer normalization if enabled
        if self.normalize_output:
            z = self.layer_norm(z)
        
        # Sample from recognition distribution if enabled
        entropy = 0
        if self.sample_rec:
            # Add small noise for sampling
            z = z + torch.randn_like(z) * 0.1
            # Compute entropy (constant for Gaussian)
            entropy = 0.5 * torch.log(2 * torch.pi * torch.tensor(0.01)) * z.shape[1]
        
        # Compute smoothness loss
        smoothness_loss = self.compute_smoothness_loss(z)
        
        # Add smoothness loss to entropy (which is used in the total loss)
        entropy = entropy + self.smoothness_weight * smoothness_loss
        
        return z, entropy