import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_
import math
import os
from torch.nn.init import uniform_, xavier_uniform_

from random import randint
from torch import optim
import copy

import json
import os
from datetime import datetime

import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N=N
        self.input_dim=input_dim
        
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B=nn.Parameter(torch.randn(self.N, self.M)*0.1)
        # External input interaction matrix C
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)
        # Readout matrix D
        self.D = nn.Parameter(torch.randn(self.N, self.M) * 0.1)
    
    def forward(self, x, m):

        batch_size, T = x.shape

        # Initialize hidden state `z`
        z = self.init_uniform((batch_size, self.M))

        # Process each time step
        for t in range(T):
            # Combine the time series value and mask into a single input
            combined_input = torch.stack([x[:, t], m[:, t]], dim=1)

            # Keep a copy of z_unactivated before applying ReLU
            z_unactivated = torch.clone(z)
            A_z_unactivated = torch.zeros_like(z_unactivated)

            if self.L > 0:
            
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)

            else:
                z_activated = z
            # Update hidden state using combined input (x[t], m[t]) and matrix C
            z = A_z_unactivated + z_activated @ self.W.t() + combined_input @ self.C.t() + self.h

        # The output is generated from the last hidden state
        output = z @ self.D.t()

        # Return the output at the last time step as the prediction
        return output

    
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
    

def train_model(
    model, 
    train_loader, 
    test_loader,
    T=10, # Sequence length, potentially needed for regularization
    num_epochs=100, 
    lr=0.001, 
    regularization=True,
    tau=1., 
    M_reg=10,
    eval_every=10 # Evaluate every 10 epochs by default
):
    """
    Train the PLRNN model on the addition problem using DataLoaders.
    
    Args:
        model: PLRNN model instance
        train_loader: DataLoader for training data (should yield x_batch, m_batch, y_batch)
        test_loader: DataLoader for test data (should yield x_batch, m_batch, y_batch)
        T: Sequence length (used for regularization scaling)
        num_epochs: Number of training epochs
        lr: Learning rate
        regularization: Whether to apply regularization
        tau: Regularization strength
        M_reg: Number of units to apply regularization to
        eval_every: Evaluate and print metrics every N epochs
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Initialize metrics tracking
    epoch_train_losses = []
    epoch_test_losses = [] # Stores test loss only on evaluation epochs
    
    print("Epoch | Avg Train Loss | Avg Test Loss")
    print("-" * 40)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        # --- Training Loop using DataLoader ---
        for x_batch, m_batch, y_batch in train_loader:
            # Ensure data is on the correct device (optional, depends on setup)
            # x_batch, m_batch, y_batch = x_batch.to(device), m_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Assuming model's forward takes x and m
            output = model(x_batch, m_batch) 
            loss = loss_fn(output.squeeze(), y_batch)
            
            if regularization:
                reg_loss=regularization_loss(model, tau, M_reg)
                loss += reg_loss
                
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        epoch_train_losses.append(avg_train_loss)
        
        avg_test_loss = float('nan') # Default if not evaluated this epoch
        # --- Evaluation Loop using DataLoader --- 
        if (epoch + 1) % eval_every == 0:
            model.eval()
            total_test_loss = 0
            num_test_batches = 0
            with torch.no_grad():
                for x_batch_test, m_batch_test, y_batch_test in test_loader:
                    # x_batch_test, m_batch_test, y_batch_test = x_batch_test.to(device), m_batch_test.to(device), y_batch_test.to(device)
                    test_output = model(x_batch_test, m_batch_test)
                    test_loss = loss_fn(test_output.squeeze(), y_batch_test)
                    total_test_loss += test_loss.item()
                    num_test_batches += 1
            
            avg_test_loss = total_test_loss / num_test_batches if num_test_batches > 0 else 0
            epoch_test_losses.append(avg_test_loss)
            
            print(f"{epoch+1:5d} | {avg_train_loss:14.4f} | {avg_test_loss:13.4f}")
            model.train() # Switch back to training mode
        else:
             epoch_test_losses.append(float('nan')) # Record NaN for non-evaluation epochs


    # Return final metrics (using the last computed average losses)
    # Find the last non-NaN test loss for the summary metric
    last_valid_test_loss = next((loss for loss in reversed(epoch_test_losses) if not math.isnan(loss)), None)

    metrics = {
        'train_loss': epoch_train_losses[-1], # Last epoch's average train loss
        'test_loss': last_valid_test_loss,    # Last calculated average test loss
    }
    
    # Returning the model trained up to the last epoch
    return model, metrics

def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on the addition task using the given dataset loader.

    Args:
        model: PLRNN model instance.
        data_loader: DataLoader for the dataset to evaluate (e.g., test_loader).
                       Expected to yield (x_batch, m_batch, y_batch).
        criterion: The loss function (e.g., nn.MSELoss).

    Returns:
        dict: A dictionary containing the average loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_batch, m_batch, y_batch in data_loader:
            # Optional: Move data to the correct device
            # x_batch, m_batch, y_batch = x_batch.to(device), m_batch.to(device), y_batch.to(device)

            output = model(x_batch, m_batch)
            loss = criterion(output.squeeze(), y_batch)
            total_loss += loss.item()
            num_batches += 1

    model.train() # Switch back to training mode

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return {
        'loss': avg_loss
    }


def save_model_with_metrics(model, metrics, hyperparams, save_dir):
    """
    Save model state, metrics, and hyperparameters from the final training epoch 
    within a specified directory. The filename will include the run_id and a timestamp.

    Args:
        model: The trained model from the final epoch.
        metrics: Dictionary of evaluation metrics from the training run 
                 (e.g., {'train_loss': ..., 'test_loss': ..., 'train_losses': [...], 'test_losses': [...]}).
        hyperparams: Dictionary of hyperparameters used for the run (must include 'run_id').
        save_dir: The directory path where files should be saved (e.g., "saved_models/run_id_...")
    """
    # Ensure the hyperparameter-specific directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Use run_id for the base filename within the directory
    run_id = hyperparams.get('run_id', 'unknown_run')
    base_name = f"{run_id}" # Filename base is now just the run_id

    # Add timestamp and find first available filename to prevent overwrites *within* the run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 0
    while True:
        # Construct filenames including type (model/meta), timestamp, and counter
        save_stem = f"{base_name}_{timestamp}_{counter}" # e.g., run_0_20240101_120000_0
        model_path = os.path.join(save_dir, f"{save_stem}_model.pt")
        meta_path = os.path.join(save_dir, f"{save_stem}_meta.json")

        if not os.path.exists(model_path) and not os.path.exists(meta_path):
            break
        counter += 1

    # Save model state dict along with hyperparams and final metrics
    save_dict = {
        'model_state_dict': model.to('cpu').state_dict(),
        'hyperparameters': hyperparams,
        'metrics': metrics # Metrics collected during training (e.g., final losses, loss history)
    }
    torch.save(save_dict, model_path)

    # Save readable metadata separately, handling potential NaNs in metrics
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            # Replace NaNs with None for JSON compatibility
            serializable_metrics[k] = [None if isinstance(item, float) and math.isnan(item) else item for item in v]
        elif isinstance(v, torch.Tensor):
            serializable_metrics[k] = v.tolist()
        elif isinstance(v, float) and math.isnan(v):
             serializable_metrics[k] = None # Replace standalone NaN with None
        else:
            serializable_metrics[k] = v
            
    with open(meta_path, 'w') as f:
        json.dump({'hyperparameters': hyperparams, 'metrics': serializable_metrics}, f, indent=4)

    print(f"Saved run results to directory: {save_dir}")
    print(f" -> Model: {os.path.basename(model_path)}")
    print(f" -> Meta:  {os.path.basename(meta_path)}")

    return model_path, meta_path