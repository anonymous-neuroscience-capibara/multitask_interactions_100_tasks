import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson, NegativeBinomial
import math

class BaseDecoder(nn.Module):
    """Base class for all decoders with unified interface"""
    def __init__(self, dz, dq):
        super().__init__()
        self.dz = dz  # latent dimension
        self.dq = dq  # output dimension (number of neurons)
        
    def forward(self, z):
        """Convert latent states to distribution parameters"""
        raise NotImplementedError
        
    def log_likelihood(self, x, z):
        """Compute log likelihood of observations given latent states"""
        raise NotImplementedError
        
    def predict_spikes(self, z, n_samples=1):
        """Generate spike predictions from latent states"""
        raise NotImplementedError

class Decoder_Poisson(BaseDecoder):
    def __init__(self, dz, dq):
        super().__init__(dz, dq)
        self.rate_net = nn.Sequential(
            nn.Linear(dz, dq),
            nn.Softplus()  # Ensure positive rates
        )
        
    def forward(self, z):
        rates = self.rate_net(z)
        return {'rates': rates}
    
    def log_likelihood(self, x, z):
        params = self.forward(z)
        rates = params['rates']
        # Compute log likelihood using Poisson PMF
        log_lik = x * torch.log(rates) - rates - torch.lgamma(x + 1)
        return log_lik.sum(dim=-1)  # Sum over neurons
    
    def predict_spikes(self, z, n_samples=1):
        params = self.forward(z)
        rates = params['rates']
        
        # Generate samples
        if n_samples == 1:
            spikes = torch.poisson(rates)
            return spikes.unsqueeze(0), rates  # Add sample dimension
        else:
            spikes = torch.stack([torch.poisson(rates) for _ in range(n_samples)])
            return spikes, rates

class Decoder_GeneralizedPoisson(BaseDecoder):
    def __init__(self, dz, dq):
        super().__init__(dz, dq)
        # Network to predict both rate and dispersion parameters
        self.param_net = nn.Sequential(
            nn.Linear(dz, 2 * dq),  # Output both rate and dispersion for each neuron
            nn.Softplus()  # Ensure positive parameters
        )
        
    def forward(self, z):
        params = self.param_net(z)
        rates = params[..., :self.dq]
        # Dispersion parameter should be in (-1, 1) for valid GP
        # Use a smaller range to ensure numerical stability
        dispersion = torch.tanh(params[..., self.dq:]) * 0.5  # Scale to (-0.5, 0.5)
        return {'rates': rates, 'dispersion': dispersion}
    
    def log_likelihood(self, x, z):
        params = self.forward(z)
        rates = params['rates']
        dispersion = params['dispersion']
        
        # Ensure rates are positive and dispersion is in valid range
        rates = torch.clamp(rates, min=1e-6)
        dispersion = torch.clamp(dispersion, min=-0.5, max=0.5)
        
        # Compute log likelihood using Generalized Poisson PMF
        # GP(x|λ,θ) = λ(λ + θx)^(x-1) * exp(-(λ + θx)) / x!
        # where λ is rate and θ is dispersion
        # Break down computation into steps for numerical stability
        
        # 1. Compute log(λ)
        log_rate = torch.log(rates)
        
        # 2. Compute log(λ + θx) safely
        rate_plus_disp = rates + dispersion * x
        # Ensure this term is positive
        rate_plus_disp = torch.clamp(rate_plus_disp, min=1e-6)
        log_rate_plus_disp = torch.log(rate_plus_disp)
        
        # 3. Compute (x-1) * log(λ + θx)
        # Handle x=0 case separately to avoid -inf
        x_minus_1 = torch.clamp(x - 1, min=0)  # Ensure non-negative
        log_term = x_minus_1 * log_rate_plus_disp
        
        # 4. Compute -(λ + θx)
        neg_rate_plus_disp = -(rates + dispersion * x)
        
        # 5. Compute -log(x!)
        # Use log gamma for numerical stability
        log_factorial = torch.lgamma(x + 1)
        
        # Combine all terms
        log_lik = (x * log_rate + 
                  log_term +
                  neg_rate_plus_disp -
                  log_factorial)
        
        # Handle any remaining numerical issues
        log_lik = torch.nan_to_num(log_lik, nan=0.0, posinf=0.0, neginf=0.0)
        
        return log_lik.sum(dim=-1)  # Sum over neurons
    
    def predict_spikes(self, z, n_samples=1):
        params = self.forward(z)
        rates = params['rates']
        dispersion = params['dispersion']
        
        # Ensure parameters are in valid ranges
        rates = torch.clamp(rates, min=1e-6)
        dispersion = torch.clamp(dispersion, min=-0.5, max=0.5)
        
        # For sampling, we use a rejection sampling approach
        # This is a simple implementation - could be optimized
        def sample_gp(rates, dispersion, n_samples):
            # Use Negative Binomial as proposal distribution
            # This is an approximation - could be improved
            p = 1 / (1 + rates * (1 + dispersion))
            r = rates * (1 + dispersion)
            nb = NegativeBinomial(r, p)
            samples = nb.sample((n_samples,))
            return samples
        
        if n_samples == 1:
            spikes = sample_gp(rates, dispersion, 1).squeeze(0)
            return spikes.unsqueeze(0), rates  # Add sample dimension
        else:
            spikes = sample_gp(rates, dispersion, n_samples)
            return spikes, rates

class Decoder_NegativeBinomial(BaseDecoder):
    def __init__(self, dz, dq):
        super().__init__(dz, dq)
        self.param_net = nn.Sequential(
            nn.Linear(dz, 2 * dq),  # Output both mean and dispersion for each neuron
            nn.Softplus()  # Ensure positive parameters
        )
        
    def forward(self, z):
        params = self.param_net(z)
        means = params[..., :self.dq]
        # Add small constant to dispersion to ensure numerical stability
        dispersion = params[..., self.dq:] + 1e-6
        return {'means': means, 'dispersion': dispersion}
    
    def log_likelihood(self, x, z):
        params = self.forward(z)
        means = params['means']
        dispersion = params['dispersion']
        
        # Compute log likelihood using Negative Binomial PMF
        nb = NegativeBinomial(dispersion, dispersion / (means + dispersion))
        log_lik = nb.log_prob(x)
        return log_lik.sum(dim=-1)  # Sum over neurons
    
    def predict_spikes(self, z, n_samples=1):
        params = self.forward(z)
        means = params['means']
        dispersion = params['dispersion']
        
        # Generate samples using Negative Binomial distribution
        nb = NegativeBinomial(dispersion, dispersion / (means + dispersion))
        if n_samples == 1:
            spikes = nb.sample()
            return spikes.unsqueeze(0), means  # Add sample dimension
        else:
            spikes = nb.sample((n_samples,))
            return spikes, means

def get_decoder(decoder_type, dz, dq):
    """Factory function to get decoder of specified type"""
    decoders = {
        'poisson': Decoder_Poisson,
        'generalized_poisson': Decoder_GeneralizedPoisson,
        'negative_binomial': Decoder_NegativeBinomial
    }
    if decoder_type not in decoders:
        raise ValueError(f"Unknown decoder type: {decoder_type}. Choose from {list(decoders.keys())}")
    return decoders[decoder_type](dz, dq) 