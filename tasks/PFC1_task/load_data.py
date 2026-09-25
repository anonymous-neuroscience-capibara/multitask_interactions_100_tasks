import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from typing import List, Optional, Dict, Any
import torch.nn.utils.rnn as rnn_utils

class TrialDataset(Dataset):
    """
    Dataset class for trial data with variable-length spike counts, external inputs (optional), and labels.
    Each trial can have different lengths, but will be padded to the maximum length in each batch.
    """
    def __init__(self, spike_counts: List[np.ndarray], external_inputs: Optional[List[np.ndarray]] = None, 
                 labels: Optional[np.ndarray] = None):
        """
        Args:
            spike_counts: List of numpy arrays, each of shape (T_i, n_neurons) where T_i can vary
            external_inputs: List of numpy arrays, each of shape (T_i, n_inputs) or None
            labels: numpy array of shape (n_trials,) containing 0/1 labels
        """
        # Store data dimensions
        self.n_trials = len(spike_counts)
        self.n_neurons = spike_counts[0].shape[1]
        self.use_inputs = external_inputs is not None
        self.n_inputs = external_inputs[0].shape[1] if self.use_inputs else 0
        
        # Store original lengths for each trial
        self.sequence_lengths = torch.tensor([len(s) for s in spike_counts])
        
        # Convert data to PyTorch tensors (as lists of tensors)
        self.spikes = [torch.FloatTensor(s) for s in spike_counts]  # List of (T_i, n_neurons)
        self.inputs = [torch.FloatTensor(i) for i in external_inputs] if self.use_inputs else None  # List of (T_i, n_inputs)
        self.labels = torch.FloatTensor(labels) if labels is not None else None  # (n_trials,)
        self.trial_ids = torch.arange(self.n_trials)  # (n_trials,)
        
        print(f"Dataset initialized with:")
        print(f"- Number of trials: {self.n_trials}")
        print(f"- Number of neurons: {self.n_neurons}")
        print(f"- Using external inputs: {self.use_inputs}")
        if self.use_inputs:
            print(f"- Number of external inputs: {self.n_inputs}")
        print(f"- Sequence lengths - min: {min(self.sequence_lengths)}, max: {max(self.sequence_lengths)}, mean: {float(self.sequence_lengths.float().mean()):.1f}")
    
    def __len__(self):
        return self.n_trials
    
    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
        - spikes: spike counts for the trial (T_i, n_neurons)
        - inputs: external inputs for the trial (T_i, n_inputs) if use_inputs=True
        - label: binary label for the trial (scalar) if labels were provided
        - trial_id: explicit trial ID (scalar)
        - length: original sequence length (scalar)
        """
        item = {
            'spikes': self.spikes[idx],      # (T_i, n_neurons)
            'trial_id': self.trial_ids[idx],  # scalar
            'length': self.sequence_lengths[idx]  # scalar
        }
        if self.use_inputs:
            item['inputs'] = self.inputs[idx]  # (T_i, n_inputs)
        if self.labels is not None:
            item['label'] = self.labels[idx]   # scalar
        return item

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    
    Args:
        batch: List of dictionaries from __getitem__
    
    Returns:
        Dictionary with padded tensors and other batch information
    """
    # Get the maximum sequence length in this batch
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    
    # Get dimensions from first item
    n_neurons = batch[0]['spikes'].shape[1]
    n_inputs = batch[0]['inputs'].shape[1] if batch[0].get('inputs') is not None else None
    
    # Initialize padded tensors
    padded_spikes = torch.zeros(batch_size, max_len, n_neurons)
    padded_inputs = torch.zeros(batch_size, max_len, n_inputs) if n_inputs is not None else None
    labels = []
    trial_ids = []
    lengths = []
    
    # Fill in the padded tensors
    for i, item in enumerate(batch):
        # Get the actual length of this sequence
        seq_len = item['length']
        
        # Copy the actual data into the padded tensor
        padded_spikes[i, :seq_len, :] = item['spikes']
        if padded_inputs is not None:
            padded_inputs[i, :seq_len, :] = item['inputs']
        
        # Collect other information
        if item.get('label') is not None:
            labels.append(item['label'])
        trial_ids.append(item['trial_id'])
        lengths.append(seq_len)
    
    # Create result dictionary
    result = {
        'spikes': padded_spikes,  # (batch_size, max_len, n_neurons)
        'trial_id': torch.stack(trial_ids),    # (batch_size,)
        'length': torch.stack(lengths)         # (batch_size,)
    }
    
    if padded_inputs is not None:
        result['inputs'] = padded_inputs  # (batch_size, max_len, n_inputs)
    if labels:
        result['label'] = torch.stack(labels)  # (batch_size,)
    
    return result

def get_dataloader(dataset: TrialDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader with the custom collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, can be increased for speed
    )

def load_and_create_dataset(session_dir: str, session_name: str, bin_width: float = 0.025,
                          use_inputs: bool = True, batch_size: int = 32, shuffle: bool = True):
    """
    Load session data from pickled lists and create dataset and dataloader.
    
    Args:
        session_dir: Directory containing the session data
        session_name: Name of the session (e.g., 'CR12B_day4')
        bin_width: Width of time bins in seconds (e.g., 0.025)
        use_inputs: Whether to load and use external inputs
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
    """
    try:
        # Construct file paths
        bin_str = f"{bin_width:.3f}".replace('.', '_')
        spikes_file = os.path.join(session_dir, f"{session_name}_spikes_{bin_str}_list.pkl")
        labels_file = os.path.join(session_dir, f"{session_name}_labels.pkl")
        
        # Load raw data
        spike_counts = joblib.load(spikes_file)  # List of (n_units, T_i) arrays
        labels = joblib.load(labels_file)  # (n_trials,)
        
        # Transpose spike counts to get (T_i, n_units)
        spike_counts = [s.T for s in spike_counts]  # Now each array is (T_i, n_units)
        
        # Load inputs if requested
        external_inputs = None
        if use_inputs:
            inputs_file = os.path.join(session_dir, f"{session_name}_inputs_{bin_str}_list.pkl")
            external_inputs = joblib.load(inputs_file)  # List of (n_inputs, T_i) arrays
            external_inputs = [i.T for i in external_inputs]  # Now each array is (T_i, n_inputs)
        
        print(f"Loaded {len(spike_counts)} trials")
        print(f"Spike counts - first trial shape: {spike_counts[0].shape} (time_steps, n_units)")
        if use_inputs:
            print(f"External inputs - first trial shape: {external_inputs[0].shape} (time_steps, n_inputs)")
        
        # Create dataset and dataloader
        dataset = TrialDataset(spike_counts, external_inputs, labels)
        dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataset, dataloader
        
    except Exception as e:
        print(f"Error loading session data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    session_dir = "Data"
    session_name = "CR12B_day4"
    bin_width = 0.025
    use_inputs = True
    
    # Load data and create dataset/dataloader
    dataset, dataloader = load_and_create_dataset(
        session_dir, 
        session_name, 
        bin_width=bin_width,
        use_inputs=use_inputs
    )
    
    # Test a batch
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"Spikes: {batch['spikes'].shape}")
    if dataset.use_inputs:
        print(f"Inputs: {batch['inputs'].shape}")
    if dataset.labels is not None:
        print(f"Labels: {batch['label'].shape}")
    print(f"Trial IDs: {batch['trial_id'].shape}")
    print(f"Sequence lengths: {batch['length'].shape}")