import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_contextual_integration_dataset(T, N_total, noise_std=0.2, invert_prob=0.5):
    """
    Generate a dataset for the context-modulated evidence integration task.
    
    Parameters:
    - T: Total sequence length (including context, evidence, and recall cue)
    - N_total: Number of samples to generate
    - noise_std: Standard deviation of noise added to evidence values
    - invert_prob: Probability that a given sample uses the inverted context
    
    Returns:
    - x_full: Input tensor of shape (N_total, T, 4) 
              (2 context dims, 1 evidence dim, 1 recall flag)
    - y_full: Target tensor of shape (N_total,), containing binary decisions (0 or 1)
    """
    # Allocate input tensor: [context_onehot(2), evidence(1), recall_flag(1)]
    x_full = torch.zeros(N_total, T, 4)
    y_full = torch.zeros(N_total)

    for i in range(N_total):
        # Choose context: 0 = normal (C1), 1 = inverted (C2)
        is_inverted = torch.rand(1).item() < invert_prob
        context_onehot = torch.tensor([0.0, 1.0]) if is_inverted else torch.tensor([1.0, 0.0])
        
        # Set context token at t=0
        x_full[i, 0, :2] = context_onehot

        # Generate noisy evidence stream (±1 + noise)
        evidence = torch.randint(0, 2, (T - 2,)) * 2 - 1  # Random ±1
        evidence = evidence.float() + noise_std * torch.randn(T - 2)

        # Fill in evidence from t=1 to T-2
        x_full[i, 1:T-1, 2] = evidence
        full_context=False
        if full_context:
            x_full[i, 1:T-1, :2] = context_onehot.repeat(T - 2, 1)
        x_full[i, 0:T-1, 3] = 0.0  # recall flag off

        # Recall cue at final timestep
        x_full[i, T-1, :2] = context_onehot
        x_full[i, T-1, 3] = 1.0  # recall flag on

        # Target is based on integrated evidence (sum)
        integrated_evidence = torch.sum(evidence)
        if is_inverted:
            integrated_evidence = -integrated_evidence

        y_full[i] = 1.0 if integrated_evidence > 0 else 0.0  # Decision: LEFT or RIGHT

    return x_full, y_full


def create_datasets(T, num_train=1000, num_test=200, num_val=200, noise_std=0.01, invert_prob=0.5):
    """
    Create train, validation, and test datasets for the contextual memory task.
    
    Args:
        T (int): Sequence length
        num_train (int): Number of training samples
        num_test (int): Number of test samples
        num_val (int): Number of validation samples
        noise_std (float): Standard deviation of noise
        invert_prob (float): Probability of inverting context
        
    Returns:
        train_dataset, val_dataset, test_dataset (TensorDataset): PyTorch datasets
    """
    # Generate datasets
    train_data, train_targets = generate_contextual_integration_dataset(T, num_train, noise_std, invert_prob)
    val_data, val_targets = generate_contextual_integration_dataset(T, num_val, noise_std, invert_prob)
    test_data, test_targets = generate_contextual_integration_dataset(T, num_test, noise_std, invert_prob)
    
    # Convert to PyTorch datasets
    # Convert targets to long type for classification
    train_dataset = TensorDataset(torch.FloatTensor(train_data), train_targets.long())
    val_dataset = TensorDataset(torch.FloatTensor(val_data), val_targets.long())
    test_dataset = TensorDataset(torch.FloatTensor(test_data), test_targets.long())
    
    return train_dataset, val_dataset, test_dataset

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        train_dataset (TensorDataset): Training dataset
        val_dataset (TensorDataset): Validation dataset
        test_dataset (TensorDataset): Test dataset
        batch_size (int): Batch size for training
        
    Returns:
        train_loader, val_loader, test_loader (DataLoader): PyTorch data loaders
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader 