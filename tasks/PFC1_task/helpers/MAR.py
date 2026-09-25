import torch
import torch.nn as nn
from hierarchical_al_rnn import HierarchicalAL_RNN

def regularization_loss(model, trial_ids, tau, M_reg):
    """
    Compute regularization loss for model parameters.
    Handles both standard AL_RNN and hierarchical HierarchicalAL_RNN models.
    For hierarchical models, computes loss for each trial's specific parameters.
    
    Args:
        model: Either AL_RNN or HierarchicalAL_RNN instance
        trial_ids: Tensor of trial IDs (batch_size,) for getting trial-specific parameters
        tau: Regularization strength
        M_reg: Number of dimensions to regularize
    """
    # Check if model is hierarchical by checking its class
    if isinstance(model, HierarchicalAL_RNN):
        # Get parameters through projections for all trials in batch
        A, W, _, _ = model.get_model_params(trial_ids)  # (batch_size, M) and (batch_size, M, M)
        
        # Compute regularization loss for each trial
        batch_loss = 0
        for i in range(M_reg):
            # Get diagonal effective weights for each trial
            diag_eff = W[:, i, i] + A[:, i]  # (batch_size,)
            batch_loss += torch.abs(diag_eff).mean()  # Average across batch
        
        return tau * batch_loss
    else:
        # For standard model, use parameters directly
        A = model.A
        W = model.W
        
        # Compute regularization loss (same for all trials)
        loss = 0
        for i in range(M_reg):
            diag_eff = W[i, i] + A[i]
            loss += torch.abs(diag_eff)
        
        return tau * loss




def regularization_loss_old(model, tau, M_reg):
    """
    Compute the regularization loss for the given model. Regularizes the first M_reg units per default.
    Args:
        model: The model object with attributes M, L, W, A, h.
        tau: Regularization strength (float).
        M_reg: Number of units to regularize (int).
    Returns:
        The combined regularization loss (torch scalar).
    """
    M, P = model.M, model.P
    linear = M - P
    loss = 0.0
    for i in range(M_reg):
        if i < linear:                        # linear unit
            diag_eff = model.W[i, i]
        else:                                 # nonlinear unit
            diag_eff = model.W[i, i] + model.A[i - linear]
        loss += (diag_eff - 1) ** 2           # first sum
        # off-diagonal W
        loss += torch.sum(model.W[i, : ]**2) - model.W[i, i]**2

        # bias term
        loss += model.h[i]**2
    return tau * loss



