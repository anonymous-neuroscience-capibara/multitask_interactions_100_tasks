import torch
import torch.nn as nn
import torch.nn.functional as F
from plrnn import AL_RNN, predict_sequence_using_gtf as predict_sequence_standard
from hierarchical_al_rnn import HierarchicalAL_RNN, predict_sequence_using_gtf as predict_sequence_hierarchical
from decoders import get_decoder
from encoder import StackedConvolutions
from hierarchical_al_rnn import latent_likelihood
from helpers.MAR import regularization_loss


class FullModel(nn.Module):
    def __init__(self, M, P, N, n_trials, input_dim, decoder_type='poisson', 
                 use_hierarchical=False, N_feat=20, fix_R_z=False, use_inputs=True, use_regularization=False, tau=0.01, M_reg=10, learn_initial_states=True):
        """
        Full model combining RNN (hierarchical or standard) with encoder and decoder.
        
        Args:
            M: Latent dimension
            P: Number of positive units
            N: Number of neurons
            n_trials: Number of trials
            input_dim: Dimension of external inputs
            decoder_type: Type of decoder ('poisson' or other)
            use_hierarchical: Whether to use hierarchical RNN
            N_feat: Number of features for hierarchical RNN
            fix_R_z: Whether to fix R_z parameter
            use_inputs: Whether to use external inputs
            use_regularization: Whether to use regularization
            tau: Regularization strength
            M_reg: Regularization dimension
            learn_initial_states: Whether to learn initial states or fix them to zeros
        """
        super(FullModel, self).__init__()
        self.M = M
        self.P = P
        self.N = N
        self.n_trials = n_trials
        self.input_dim = input_dim if use_inputs else 0
        self.use_inputs = use_inputs
        self.use_hierarchical = use_hierarchical
        self.N_feat = N_feat if use_hierarchical else None
        self.learn_initial_states = learn_initial_states

        
        # Initialize RNN (either hierarchical or standard)
        if use_hierarchical:
            self.rnn = HierarchicalAL_RNN(
                M=self.M, 
                P=self.P, 
                N_feat=self.N_feat,
                n_trials=self.n_trials,
                input_dim=self.input_dim,
                use_inputs=use_inputs,
                learn_initial_states=learn_initial_states
            )
            self.predict_sequence = predict_sequence_hierarchical
        else:
            self.rnn = AL_RNN(
                M=self.M, 
                P=self.P, 
                input_dim=self.input_dim,
                n_trials=self.n_trials,
                use_inputs=use_inputs,
                learn_initial_states=learn_initial_states
            )
            self.predict_sequence = predict_sequence_standard
        
        # Initialize other components
        self.decoder = get_decoder(decoder_type, dz=self.M, dq=N)
        self.encoder = StackedConvolutions(dim_x=N, dim_z=self.M, sample_rec=False)
        
        # Initialize R_z as a learnable parameter
        self.register_buffer('R_z', torch.ones(self.M))
        self.R_z_param = nn.Parameter(torch.ones(self.M))
        self.fix_R_z = fix_R_z
        
    
    def forward(self, batch, trial_ids, alpha, n_interleave, beta_pred, beta_enc, beta_cons, beta_class=0.1, beta_ent=0.1, use_regularization=False, tau=0.01, M_reg=10):
        """
        Forward pass of the full model.
        
        Args:
            batch: Dictionary containing:
                - spikes: Spike counts (batch_size, T, N)
                - inputs: External inputs (batch_size, T, input_dim) if use_inputs=True
                - label: Binary labels (batch_size,)
                - length: Sequence lengths (batch_size,)
            trial_ids: Tensor of trial IDs (batch_size,)
            alpha: Teacher forcing parameter
            n_interleave: Teacher forcing frequency
            beta_pred: Weight for prediction loss
            beta_enc: Weight for encoder loss
            beta_cons: Weight for consistency loss
            beta_class: Weight for classification loss
        """
        # Update R_z from parameter if not fixed
        if not self.fix_R_z:
            self.R_z.copy_(self.R_z_param)
        
        # Get data from batch
        spikes = batch['spikes']  # (batch_size, T, N)
        inputs = batch['inputs'] if self.use_inputs else None  # (batch_size, T, input_dim) or None
        lengths = batch['length']  # (batch_size,)
        
        # Get initial states and generate sequence
        initial_states = self.rnn.get_initial_state(trial_ids)
        encoded_x, entropy = self.encoder(spikes)
        
        # Use appropriate predict_sequence function based on model type
        if self.use_hierarchical:
            z_lat, logits = self.predict_sequence(
                self.rnn, initial_states, inputs,
                encoded_x, alpha, n_interleave, trial_ids, lengths
            )
        else:
            z_lat, logits = self.predict_sequence(
                self.rnn, initial_states, inputs,
                encoded_x, alpha, n_interleave, lengths
            )
        
        # Compute losses
        log_lik_pred = self.decoder.log_likelihood(spikes, z_lat)
        log_lik_enc = self.decoder.log_likelihood(spikes, encoded_x)
        kl_div = latent_likelihood(z_lat, encoded_x, self.R_z)
        
        # Reduce to scalars
        log_lik_pred = log_lik_pred.mean()
        log_lik_enc = log_lik_enc.mean()
        kl_div = kl_div.mean()
        
        # Compute classification loss using cross entropy with raw logits
        classification_loss = F.cross_entropy(
            logits,  # (batch_size, 2) - raw logits
            batch['label'].long(),  # (batch_size,) - class indices (0 or 1)
            reduction='mean'
        )
        
        # Combine losses
        loss_pred = -beta_pred * log_lik_pred
        loss_enc = -beta_enc * log_lik_enc
        loss_cons = -beta_cons * kl_div
        loss_class = beta_class * classification_loss
        loss_ent = beta_ent * entropy  # Entropy loss (currently disabled)
        
        total_loss = loss_pred + loss_enc + loss_cons + loss_class + loss_ent

        if use_regularization:
            total_loss += regularization_loss(self.rnn, trial_ids, tau, M_reg)
        
        # Get probabilities for accuracy computation
        probs = F.softmax(logits, dim=1)
        
        return {
            'total_loss': total_loss,
            'prediction_loss': loss_pred.item(),
            'encoder_loss': loss_enc.item(),
            'consistency_loss': loss_cons.item(),
            'classification_loss': loss_class.item(),
            'entropy_loss': loss_ent,
            'predictions': probs[:, 1]  # Return probability for class 1 for accuracy computation
        }
    
    def fix_R_z_gradients(self, fix=True):
        """Toggle whether R_z should be updated during training"""
        self.fix_R_z = fix
        if fix:
            self.R_z_param.requires_grad_(False)
        else:
            self.R_z_param.requires_grad_(True)
    
    @torch.no_grad()
    def predict(self, batch, trial_ids=None, alpha=0.0, n_interleave=1, n_samples=1):
        """
        Generate spike predictions from input data
        
        Args:
            batch: Dictionary containing:
                - spikes: Spike counts (batch_size, T, N)
                - inputs: External inputs (batch_size, T, input_dim) if use_inputs=True
            trial_ids: Tensor of trial IDs (batch_size,) (if None, uses all trials)
            alpha: Teacher forcing parameter
            n_interleave: Teacher forcing interval
            n_samples: Number of samples to generate
        """
        self.eval()
        
        spikes = batch['spikes']
        inputs = batch['inputs'] if self.use_inputs else None
        
        if trial_ids is None:
            trial_ids = torch.arange(spikes.shape[0], device=spikes.device)
        initial_states = self.rnn.get_initial_state(trial_ids)
        
        encoded_x, _ = self.encoder(spikes)
        
        # Use appropriate predict_sequence function based on model type
        if self.use_hierarchical:
            z_hat, _ = self.predict_sequence(
                self.rnn, initial_states, inputs,
                encoded_x, alpha, n_interleave, trial_ids
            )
        else:
            z_hat, _ = self.predict_sequence(
                self.rnn, initial_states, inputs,
                encoded_x, alpha, n_interleave
            )
        
        spikes, rates = self.decoder.predict_spikes(z_hat, n_samples)
        return spikes, rates, z_hat 