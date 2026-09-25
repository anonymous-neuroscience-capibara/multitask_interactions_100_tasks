import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_full_dataset(T, N_total, addition):

    x_full = torch.rand(N_total, T)  # Random numbers between 0 and 1
    m_full = torch.zeros(N_total, T)  # Binary mask initialized to zero
    
    # Assign two distinct random indices in the first half of the sequence
    for i in range(N_total):
        idx = torch.randperm(T // 2)[:2]  # Ensure two distinct indices in the first half
        m_full[i, idx] = 1  # Set mask to 1 at the two random positions in the first half
    
    # The target is the sum of the two values marked by the mask
    if addition:
        y_full = torch.sum(x_full * m_full, dim=1)
    else:
        y_full = torch.prod(x_full * m_full + (1 - m_full), dim=1)
        y_full=-torch.log(y_full)
    return x_full, m_full, y_full


def create_datasets(seq_len, N_total, num_train=1000, num_test=200):
    """
    Create train and test datasets for the copy task.
    """
    # Generate training data
    x_train, m_train, y_train = generate_full_dataset(
        T=seq_len,
        N_total=num_train,
        addition=True
    )
    
    # Generate test data
    x_test, m_test,y_test = generate_full_dataset(
        T=seq_len,
        N_total=num_test,
        addition=True
    )
    
    # Create data loaders
    train_dataset = TensorDataset(x_train, m_train, y_train)
    test_dataset = TensorDataset(x_test, m_test, y_test)
    
    return train_dataset, test_dataset


