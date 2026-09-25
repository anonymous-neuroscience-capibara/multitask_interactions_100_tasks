import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_copy_task_dataset_discrete(seq_len=10, num_symbols=8, num_samples=10000, delay=0, include_time_encoding=True, time_dim=16):
    def get_sinusoidal_encoding(timesteps, dim):
        position = torch.arange(timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe = torch.zeros(timesteps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    input_seqs = []
    target_seqs = []

    total_len = seq_len + 1 + delay + seq_len

    # Create sinusoidal positional encodings once (shared across samples)
    if include_time_encoding:
        time_encoding = get_sinusoidal_encoding(total_len, time_dim)  # [T, time_dim]

    for _ in range(num_samples):
        symbols = torch.randint(1, num_symbols, size=(seq_len,))  # symbols from 1 to num_symbols-1

        # One-hot input: [T, num_symbols]
        x = torch.zeros(total_len, num_symbols)
        x[torch.arange(seq_len), symbols] = 1
        x[seq_len, 0] = 1  # cue token
        x[seq_len + 1 + delay:, 0] = 1  # dummy decoder inputs

        if include_time_encoding:
            x = torch.cat([x, time_encoding], dim=-1)  # concat on last dim

        # Create target
        y = torch.full((total_len,), -100, dtype=torch.long)
        y[seq_len + 1 + delay:] = symbols  # model should reproduce symbols here

        input_seqs.append(x)
        target_seqs.append(y)

    return torch.stack(input_seqs), torch.stack(target_seqs)



def create_datasets(seq_len, num_symbols, delay, time_dim, num_train=1000, num_test=200):
    """
    Create train and test datasets for the copy task.
    """
    # Generate training data
    x_train, y_train = generate_copy_task_dataset_discrete(
        seq_len=seq_len,
        num_symbols=num_symbols,
        num_samples=num_train,
        delay=delay,
        include_time_encoding=True,
        time_dim=time_dim
    )
    
    # Generate test data
    x_test, y_test = generate_copy_task_dataset_discrete(
        seq_len=seq_len,
        num_symbols=num_symbols,
        num_samples=num_test,
        delay=delay,
        include_time_encoding=True,
        time_dim=time_dim
    )
    
    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    return train_dataset, test_dataset


