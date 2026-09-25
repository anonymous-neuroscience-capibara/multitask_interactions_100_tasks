import torch
from torch.utils.data import Dataset, DataLoader

class TrialDataset(Dataset):
    """
    Dataset class for trial data with spike counts, external inputs, and labels.
    Each trial has:
    - 60 time steps
    - 29 neurons for spike counts
    - 8 dimensions for external inputs
    """
    def __init__(self, spike_counts, external_inputs, labels):
        """
        Args:
            spike_counts: numpy array of shape (n_trials=479, n_timepoints=60, n_neurons=29)
            external_inputs: numpy array of shape (n_trials=479, n_timepoints=60, n_inputs=8)
            labels: numpy array of shape (n_trials=479,) containing 0/1 labels
        """
        # Store data dimensions
        self.n_trials = spike_counts.shape[0]
        self.n_timepoints = spike_counts.shape[1]  # 60
        self.n_neurons = spike_counts.shape[2]     # 29
        self.n_inputs = external_inputs.shape[2]   # 8
        
        # Convert data to PyTorch tensors
        self.spikes = torch.FloatTensor(spike_counts)      # (479, 60, 29)
        self.inputs = torch.FloatTensor(external_inputs)   # (479, 60, 8)
        self.labels = torch.FloatTensor(labels)            # (479,)
        self.trial_indices = torch.arange(self.spikes.shape[0])  # 0 to 478
        
        print(f"Dataset initialized with:")
        print(f"- Number of trials: {self.n_trials}")
        print(f"- Timepoints per trial: {self.n_timepoints}")
        print(f"- Number of neurons: {self.n_neurons}")
        print(f"- Number of external inputs: {self.n_inputs}")
        print(f"- Number of correct trials: {torch.sum(self.labels == 1).item()}")
        print(f"- Number of incorrect trials: {torch.sum(self.labels == 0).item()}")
    
    def __len__(self):
        return self.spikes.shape[0]
    
    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
        - spikes: spike counts for the trial (60, 29)
        - inputs: external inputs for the trial (60, 8)
        - label: binary label for the trial (scalar)
        - trial_indices: index of the trial (scalar)
        """
        return {
            'spikes': self.spikes[idx],      # (60, 29)
            'inputs': self.inputs[idx],      # (60, 8)
            'label': self.labels[idx],       # scalar
            'trial_indices': self.trial_indices[idx]  # scalar
        }

def get_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Creates a DataLoader for the trial dataset.
    Each batch will contain:
    - spikes: (batch_size, 60, 29)
    - inputs: (batch_size, 60, 8)
    - labels: (batch_size,)
    - trial_indices: (batch_size,)
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)