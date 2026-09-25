import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
import math
import os
from torch.nn.init import uniform_, xavier_uniform_
import random
from random import randint
from torch import optim
import copy
import sys
import json
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim, nonlinearity_type="relu"):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N  # This is num_symbols
        self.input_dim = input_dim
        self.nonlinearity_type = nonlinearity_type
        
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = nn.Parameter(torch.randn(self.N, self.M)*0.1)
        # External input interaction matrix C
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)
        # Readout matrix D
        self.D = nn.Parameter(torch.randn(self.N, self.M) * 0.1)
    
    def forward(self, x):

        batch_size, T, input_dim = x.shape

        # Initialize hidden state `z`
        z = self.init_uniform((batch_size, self.M))
        outputs = []
        # Process each time step
        latents=[]
        
        # Select nonlinearity function
        if self.nonlinearity_type == "relu":
            nonlin_fn = F.relu
        elif self.nonlinearity_type == "tanh":
            nonlin_fn = torch.tanh
        elif self.nonlinearity_type == "gelu":
            nonlin_fn = F.gelu
        else:
            raise ValueError(f"Unsupported nonlinearity_type: {self.nonlinearity_type}")
        
        for t in range(T):
            # Combine the time series value and mask into a single input
            x_t = x[:, t, :]

            # Keep a copy of z_unactivated before applying nonlinearity
            z_unactivated = torch.clone(z)
            A_z_unactivated = torch.zeros_like(z_unactivated)

            if self.L > 0:
            
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], nonlin_fn(z[:, -self.L:])], dim=1)

            else:
                z_activated = z
            
            z = A_z_unactivated + z_activated @ self.W.t() + x_t @ self.C.t() + self.h
            outputs.append(z @ self.D.t())
            latents.append(z)

        # Return the output at the last time step as the prediction
        return torch.stack(outputs, dim=1), torch.stack(latents, dim=1)

    
    def initialize_AWh_random(self):
        #Randomly initialize A, W, h
        A = nn.Parameter(torch.randn(self.L) * 0.01)
        W = nn.Parameter(torch.randn(self.M, self.M)*0.01)
        h = nn.Parameter(torch.randn(self.M)*0.01)
        return A, W, h
    
    
    def init_uniform(self, shape):

        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)


class LSTM(nn.Module):
    def __init__(self, M, N, input_dim):
        super(LSTM, self).__init__()
        self.M = M
        self.N = N  # This is num_symbols
        self.input_dim = input_dim
        
        # LSTM layer with hidden size M
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=M, batch_first=True)
        
        # Output projection layer
        self.output_layer = nn.Linear(M, N)
    
    def forward(self, x):
        batch_size, T, input_dim = x.shape
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, batch_size, self.M, device=x.device)
        c0 = torch.zeros(1, batch_size, self.M, device=x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Project each time step to output dimension
        outputs = []
        for t in range(T):
            output_t = self.output_layer(lstm_out[:, t, :])
            outputs.append(output_t)
        
        # Return the sequence of outputs and the final hidden state
        return torch.stack(outputs, dim=1), lstm_out


class GRU(nn.Module):
    def __init__(self, M, N, input_dim):
        super(GRU, self).__init__()
        self.M = M
        self.N = N  # This is num_symbols
        self.input_dim = input_dim
        
        # GRU layer with hidden size M
        self.gru = nn.GRU(input_size=input_dim, hidden_size=M, batch_first=True)
        
        # Output projection layer
        self.output_layer = nn.Linear(M, N)
    
    def forward(self, x):
        batch_size, T, input_dim = x.shape
        
        # Initialize hidden state
        h0 = torch.zeros(1, batch_size, self.M, device=x.device)
        
        # GRU forward pass
        gru_out, hn = self.gru(x, h0)
        
        # Project each time step to output dimension
        outputs = []
        for t in range(T):
            output_t = self.output_layer(gru_out[:, t, :])
            outputs.append(output_t)
        
        # Return the sequence of outputs and the final hidden state
        return torch.stack(outputs, dim=1), gru_out


def train_model(
    model, 
    train_loader,
    test_loader,
    seq_len,
    num_epochs=5000,
    eval_every=50,
    lr=0.001,
    tau=0.005,
    M_reg=25,
):
    """
    Train the model (PLRNN, LSTM, or GRU) for a fixed number of epochs.
    Args:
        model: PLRNN, LSTM, or GRU model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        seq_len: Length of sequences to copy
        num_epochs: Number of training epochs
        eval_every: Evaluate on test set every N epochs
        lr: Learning rate
        tau: Global regularization strength (only for PLRNN)
        M_reg: Number of units to apply regularization to (only for PLRNN)
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Determine if regularization should be applied (only for PLRNN/AL-RNN)
    # LSTM and GRU models do not use regularization
    apply_regularization = isinstance(model, PLRNN)
    
    print("Epoch | Train Loss | Test Loss | Symbol Acc | Sequence Acc")
    print("-" * 55)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output, _ = model(x_batch)
            
            output_masked = output[:, -seq_len:, :].reshape(-1, model.N)
            y_masked = y_batch[:, -seq_len:].reshape(-1)
            
            loss = criterion(output_masked, y_masked)
            
            # Apply regularization only for PLRNN/AL-RNN models
            # LSTM and GRU models do not use regularization (tau and M_reg are ignored)
            if apply_regularization:
                reg_loss = regularization_loss(model, tau, M_reg)
                loss += reg_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluate on test set periodically
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate_model(model, test_loader, seq_len, criterion)
            train_loss = total_loss / num_batches
            
            print(f"{epoch+1:5d} | {train_loss:.4f} | {metrics['loss']:.4f} | "
                  f"{metrics['symbol_accuracy']:9.2f} | {metrics['sequence_accuracy']:11.2f}")
    
    # Perform final evaluation
    final_metrics = evaluate_model(model, test_loader, seq_len, criterion)
    print(f"\nTraining completed after {num_epochs} epochs.")
    print(f"Final sequence accuracy: {final_metrics['sequence_accuracy']:.2f}%")
    
    return model, final_metrics

def evaluate_model(model, data_loader, seq_len, criterion):
    """
    Evaluate the model on the given dataset.
    """
    model.eval()
    total_loss = 0
    total_sequences = 0
    correct_sequences = 0
    total_symbols = 0
    correct_symbols = 0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            output, _ = model(x_batch)
            
            output_masked = output[:, -seq_len:, :]
            y_masked = y_batch[:, -seq_len:]
            
            # Compute loss
            output_flat = output_masked.reshape(-1, output_masked.size(-1))
            y_flat = y_masked.reshape(-1)
            loss = criterion(output_flat, y_flat)
            total_loss += loss.item()
            
            # Compute accuracies
            predictions = output_masked.argmax(dim=-1)
            correct_symbols += (predictions == y_masked).sum().item()
            total_symbols += y_masked.numel()
            
            sequence_correct = (predictions == y_masked).all(dim=1)
            correct_sequences += sequence_correct.sum().item()
            total_sequences += len(sequence_correct)
    
    model.train()
    
    return {
        'loss': total_loss / len(data_loader),
        'symbol_accuracy': correct_symbols / total_symbols * 100,
        'sequence_accuracy': correct_sequences / total_sequences * 100
    }


def train_model_variable_delay(
    model, 
    train_loader,
    test_loader,
    seq_len,
    cue_positions_train,
    cue_positions_test,
    num_epochs=5000,
    eval_every=50,
    lr=0.001,
    tau=0.005,
    M_reg=25,
):
    """
    Train the model for the variable delay task.
    
    Args:
        model: PLRNN, LSTM, or GRU model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        seq_len: Length of sequences to remember
        cue_positions_train: List of cue positions for training data
        cue_positions_test: List of cue positions for test data
        num_epochs: Number of training epochs
        eval_every: Evaluate on test set every N epochs
        lr: Learning rate
        tau: Global regularization strength (only for PLRNN)
        M_reg: Number of units to apply regularization to (only for PLRNN)
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Determine if regularization should be applied (only for PLRNN/AL-RNN)
    apply_regularization = isinstance(model, PLRNN)
    
    print("Epoch | Train Loss | Test Loss | Symbol Acc | Sequence Acc")
    print("-" * 55)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Get batch indices for cue positions
        batch_indices = []
        for i, (x_batch, y_batch) in enumerate(train_loader):
            batch_size = x_batch.size(0)
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_cues = cue_positions_train[start_idx:end_idx]
            batch_indices.append((x_batch, y_batch, batch_cues))
        
        for x_batch, y_batch, batch_cues in batch_indices:
            optimizer.zero_grad()
            output, _ = model(x_batch)
            
            # For each sample in the batch, extract the target sequence based on cue position
            batch_loss = 0
            valid_samples = 0
            
            for i, cue_pos in enumerate(batch_cues):
                # Target sequence starts at cue_pos + 1 and has length seq_len
                recall_start = cue_pos + 1
                recall_end = recall_start + seq_len
                
                # Extract target sequence
                target_seq = y_batch[i, recall_start:recall_end]
                
                # Extract model output for the same positions
                output_seq = output[i, recall_start:recall_end, :]
                
                # Only compute loss if we have valid targets (not all -100)
                if (target_seq != -100).any():
                    loss_i = criterion(output_seq, target_seq)
                    batch_loss += loss_i
                    valid_samples += 1
            
            if valid_samples > 0:
                loss = batch_loss / valid_samples
                
                # Apply regularization only for PLRNN/AL-RNN models
                if apply_regularization:
                    reg_loss = regularization_loss(model, tau, M_reg)
                    loss += reg_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Evaluate on test set periodically
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate_model_variable_delay(model, test_loader, seq_len, cue_positions_test, criterion)
            train_loss = total_loss / max(num_batches, 1)
            
            print(f"{epoch+1:5d} | {train_loss:.4f} | {metrics['loss']:.4f} | "
                  f"{metrics['symbol_accuracy']:9.2f} | {metrics['sequence_accuracy']:11.2f}")
    
    # Perform final evaluation
    final_metrics = evaluate_model_variable_delay(model, test_loader, seq_len, cue_positions_test, criterion)
    print(f"\nTraining completed after {num_epochs} epochs.")
    print(f"Final sequence accuracy: {final_metrics['sequence_accuracy']:.2f}%")
    
    return model, final_metrics

def evaluate_model_variable_delay(model, data_loader, seq_len, cue_positions, criterion):
    """
    Evaluate the model on the variable delay task.
    """
    model.eval()
    total_loss = 0
    total_sequences = 0
    correct_sequences = 0
    total_symbols = 0
    correct_symbols = 0
    
    # Get batch indices for cue positions
    batch_indices = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        batch_size = x_batch.size(0)
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_cues = cue_positions[start_idx:end_idx]
        batch_indices.append((x_batch, y_batch, batch_cues))
    
    with torch.no_grad():
        for x_batch, y_batch, batch_cues in batch_indices:
            output, _ = model(x_batch)
            
            for i, cue_pos in enumerate(batch_cues):
                # Target sequence starts at cue_pos + 1 and has length seq_len
                recall_start = cue_pos + 1
                recall_end = recall_start + seq_len
                
                # Extract target sequence
                target_seq = y_batch[i, recall_start:recall_end]
                
                # Extract model output for the same positions
                output_seq = output[i, recall_start:recall_end, :]
                
                # Only evaluate if we have valid targets (not all -100)
                if (target_seq != -100).any():
                    # Compute loss
                    loss = criterion(output_seq, target_seq)
                    total_loss += loss.item()
                    
                    # Compute accuracies
                    predictions = output_seq.argmax(dim=-1)
                    
                    # Only count symbols that are not padding (-100)
                    valid_mask = target_seq != -100
                    if valid_mask.any():
                        correct_symbols += (predictions[valid_mask] == target_seq[valid_mask]).sum().item()
                        total_symbols += valid_mask.sum().item()
                        
                        # Check if entire sequence is correct
                        sequence_correct = (predictions[valid_mask] == target_seq[valid_mask]).all()
                        if sequence_correct:
                            correct_sequences += 1
                        total_sequences += 1
    
    model.train()
    
    return {
        'loss': total_loss / max(len(batch_indices), 1),
        'symbol_accuracy': (correct_symbols / max(total_symbols, 1)) * 100,
        'sequence_accuracy': (correct_sequences / max(total_sequences, 1)) * 100
    }


def save_model_with_metrics(model, metrics, hyperparams, save_dir):
    """
    Save model state, metrics, and hyperparameters within a specified directory.
    The filename will include the run_id and a timestamp.

    Args:
        model: The trained model (best one found).
        metrics: Dictionary of final evaluation metrics.
        hyperparams: Dictionary of hyperparameters used for the run (must include 'run_id').
        save_dir: The directory path where files should be saved (e.g., "saved_models/L50_tau0.01...")
    """
    # Ensure the hyperparameter-specific directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Use run_id for the base filename within the directory
    run_id = hyperparams.get('run_id', 'unknown_run')
    base_name = f"{run_id}" # Filename base is now just the run_id

    # Add timestamp and find first available filename to prevent overwrites *within* the run_id (e.g., if saved multiple times)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 0
    while True:
        # Construct filenames including type (model/meta), timestamp, and counter
        save_stem = f"{base_name}_{timestamp}_{counter}" # e.g., run_0_L50..._20240101_120000_0
        model_path = os.path.join(save_dir, f"{save_stem}_model.pt")
        meta_path = os.path.join(save_dir, f"{save_stem}_meta.json")

        if not os.path.exists(model_path) and not os.path.exists(meta_path):
            break
        counter += 1

    # Save model state dict
    save_dict = {
        'model_state_dict': model.to('cpu').state_dict(),
        'hyperparameters': hyperparams,
        'metrics': metrics # Metrics from the *best* model evaluation point
    }
    torch.save(save_dict, model_path)

    # Save readable metadata separately
    with open(meta_path, 'w') as f:
        serializable_metrics = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        json.dump({'hyperparameters': hyperparams, 'metrics': serializable_metrics}, f, indent=4)

    print(f"Saved run results to directory: {save_dir}")
    print(f" -> Model: {os.path.basename(model_path)}")
    print(f" -> Meta:  {os.path.basename(meta_path)}")

    return model_path, meta_path