import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from decoders import get_decoder


class AL_RNN(nn.Module):
    def __init__(self, M, P, input_dim, n_trials, use_inputs=True, learn_initial_states=True):
        """
        Args:
            M: Latent dimension
            P: Number of positive units
            input_dim: Dimension of external inputs
            n_trials: Number of trials (for initial states)
            use_inputs: Whether to use external inputs (if False, input_dim is ignored)
            learn_initial_states: Whether to learn initial states or fix them to zeros
        """
        super(AL_RNN, self).__init__()

        # Initialize model dimensions
        self.M = M
        self.P = P
        self.input_dim = input_dim if use_inputs else 0
        self.use_inputs = use_inputs
        self.learn_initial_states = learn_initial_states
        
        # Initialize model parameters
        self.A, self.W, self.h = self.initialize_AWh_random()
        if use_inputs:
            self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.5)  # Input projection
        # Simple linear projection for classification (2 classes)
        self.D = nn.Parameter(torch.randn(2, self.M) * 0.1)  # Output projection for binary classification
        
        # Initialize initial states based on learn_initial_states flag
        if learn_initial_states:
            self.z0 = nn.Parameter(torch.zeros(n_trials, self.M))  # Learnable initial states
        else:
            self.register_buffer('z0', torch.zeros(n_trials, self.M))  # Fixed zero initial states
        
        print(f"Initialized AL_RNN with:")
        print(f"- Latent dimension (M): {self.M}")
        print(f"- Number of positive units (P): {self.P}")
        print(f"- Using external inputs: {self.use_inputs}")
        if use_inputs:
            print(f"- Input dimension: {self.input_dim}")
        print(f"- Number of trials: {n_trials}")
        print(f"- Initial states: {'Learnable' if learn_initial_states else 'Fixed to zeros'}")
        print(f"- Initial states shape: {self.z0.shape}")
    
    def forward_step(self, z, input):
        """Single step forward pass of the RNN"""
        # Make a copy of the input tensor to retain unactivated values
        z_unactivated = torch.clone(z)
        
        # Apply ReLU activation on the last P units (without inplace operation)
        z_activated = torch.cat([z[:, :-self.P], F.relu(z[:, -self.P:])], dim=1)
        
        # Compute the forward pass with or without external input
        if self.use_inputs:
            return self.A * z_unactivated + z_activated @ self.W.t() + self.h + input @ self.C.t()
        else:
            return self.A * z_unactivated + z_activated @ self.W.t() + self.h
    
    def forward(self, batch, trial_ids, alpha, n_interleave, beta_pred, beta_enc, beta_cons, beta_class=0.1):
        """
        Full forward pass for training.
        
        Args:
            batch: Dictionary containing:
                - spikes: (batch_size, T, N)
                - inputs: (batch_size, T, input_dim)
                - label: (batch_size,)
            trial_ids: Tensor of trial IDs (batch_size,)
            alpha: Teacher forcing parameter
            n_interleave: Teacher forcing frequency
            beta_pred: Weight for prediction loss
            beta_enc: Weight for encoder loss
            beta_cons: Weight for consistency loss
            beta_class: Weight for classification loss
            
        Returns:
            Dictionary containing:
                - total_loss: Combined loss
                - prediction_loss: Loss for spike prediction
                - encoder_loss: Loss for encoder
                - consistency_loss: Loss for consistency
                - classification_loss: Loss for classification
                - predictions: Binary predictions
        """
        # Get initial states for the batch using trial IDs
        initial_states = self.get_initial_state(trial_ids)
        
        # Get inputs and encoded states
        inputs = batch['inputs']
        encoded_x = batch['encoded_x']  # Assuming this is provided by the encoder
        
        # Predict sequence using teacher forcing
        latent_states, predictions = predict_sequence_using_gtf(
            self, initial_states, inputs, encoded_x, alpha, n_interleave
        )
        
        # Compute losses
        prediction_loss = self.compute_prediction_loss(latent_states, batch['spikes'])
        encoder_loss = self.compute_encoder_loss(latent_states, encoded_x)
        consistency_loss = self.compute_consistency_loss(latent_states)
        classification_loss = self.compute_classification_loss(predictions, batch['label'])
        
        # Combine losses
        total_loss = (
            beta_pred * prediction_loss +
            beta_enc * encoder_loss +
            beta_cons * consistency_loss +
            beta_class * classification_loss
        )
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'encoder_loss': encoder_loss,
            'consistency_loss': consistency_loss,
            'classification_loss': classification_loss,
            'predictions': predictions
        }
    
    def predict_label(self, z):
        """Predict class logits using linear projection"""
        # Simple linear projection to 2D (one output per class)
        return z @ self.D.t()  # (batch_size, 2) - raw logits for both classes
    
    def get_initial_state(self, trial_ids):
        """Get initial states for the given trial IDs"""
        if self.learn_initial_states:
            return self.z0[trial_ids]  # Return learnable initial states
        else:
            return torch.zeros_like(self.z0[trial_ids])  # Return zeros of same shape
    
    def initialize_AWh_random(self):
        """Initialize RNN parameters"""
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.1)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h
    
    def normalized_positive_definite(self, M):
        """Generate normalized positive definite matrix"""
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)
        eigenvalues = np.linalg.eigvals(K)
        lambda_max = np.max(np.abs(eigenvalues))
        return torch.tensor(K / lambda_max).float()

@torch.no_grad()
def predict_free_sequence(model, initial_state, inputs, T):
    """
    Predict sequence without teacher forcing.
    
    Args:
        model: AL_RNN model
        initial_state: Initial latent state (batch_size, M)
        inputs: External inputs (batch_size, T, input_dim)
        T: Sequence length
        
    Returns:
        tuple: (latent_states, predicted_labels)
            - latent_states: (batch_size, T, M)
            - predicted_labels: (batch_size, 1)
    """
    b, M = initial_state.size()
    Z = torch.empty(size=(T, b, M), device=initial_state.device)
    z = initial_state

    # Predict sequence
    for t in range(T):
        z = model(z, inputs[:, t])  # Use current input
        Z[t] = z
    
    # Get final prediction
    final_states = Z[-1]  # (batch_size, M)
    predicted_labels = model.predict_label(final_states)
    
    return Z.permute(1, 0, 2), predicted_labels

def predict_sequence_using_gtf(model, initial_state, inputs, encoded_x, alpha, n_interleave, lengths=None):
    """
    Predict sequence using teacher forcing.
    
    Args:
        model: AL_RNN model
        initial_state: Initial latent state (batch_size, M)
        inputs: External inputs (batch_size, T, input_dim) or None if not using inputs
        encoded_x: Encoded states for teacher forcing (batch_size, T, M)
        alpha: Teacher forcing parameter
        n_interleave: Teacher forcing frequency
        lengths: Tensor of sequence lengths (batch_size,) or None if all sequences are same length
    """
    T = encoded_x.size(1)  # number of time steps
    b = initial_state.size(0)  # batch size
    M = initial_state.size(1)  # latent dimension
    Z = torch.empty(size=(T, b, M), device=initial_state.device)
    z = initial_state

    # Generate sequence with teacher forcing
    for t in range(T):
        # Apply teacher forcing at regular intervals
        if (t % n_interleave == 0) and (t > 0):
            z = alpha * encoded_x[:, t] + (1 - alpha) * z
        
        # Update state with or without current input using forward_step
        if model.use_inputs:
            z = model.forward_step(z, inputs[:, t])  # inputs[:, t] is (batch_size, input_dim)
        else:
            z = model.forward_step(z, None)  # No input needed
        Z[t] = z
    
    # Get final states based on actual sequence lengths if provided
    if lengths is not None:
        # Convert lengths to 0-based indices for indexing
        indices = lengths - 1  # (batch_size,)
        # Get the final state for each sequence based on its length
        final_states = Z[indices, torch.arange(b, device=Z.device)]  # (batch_size, M)
    else:
        # If no lengths provided, use the last state
        final_states = Z[-1]  # (batch_size, M)
    
    predicted_labels = model.predict_label(final_states)
    
    return Z.permute(1, 0, 2), predicted_labels  # (batch_size, T, M), (batch_size, 2)

def teacher_force(z, x, alpha):
    """Teacher force the state z towards x with strength alpha"""
    return alpha * x + (1 - alpha) * z

def latent_likelihood(z_lat, z_enc, R_z):
    """Compute latent likelihood"""
    def mahalonobis_distance(residual, matrix):
        distance = 0
        for i in range(residual.shape[0]):
            distance += -0.5 * (residual[i].t() @ residual[i] * torch.inverse(torch.diag(matrix ** 2))).sum()
        return distance

    def log_det(diagonal_matrix):
        return -torch.log(diagonal_matrix).sum()

    LL_z = mahalonobis_distance(z_lat - z_enc, R_z)
    T = z_lat.shape[1]

    return LL_z + log_det(R_z) * (T - 1) / 2




    