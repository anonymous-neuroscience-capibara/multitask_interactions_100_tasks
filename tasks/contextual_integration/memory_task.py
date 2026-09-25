import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import os
import json
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss

class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N
        self.input_dim = input_dim
        
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = nn.Parameter(torch.randn(self.N, self.M) * 0.1)
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(self.N, self.M) * 0.1)
    
    def forward(self, x):
        """
        x: [batch_size, T, input_dim] — input sequence
        Returns:
        - output: logits from last time step [batch_size, N]
        - z: latent state at final step [batch_size, M]
        """
        batch_size, T, _ = x.shape

        # Initialize hidden state z
        z = torch.zeros(batch_size, self.M, device=x.device)
        Z = []
        Z.append(z)
        
        # Process sequence
        for t in range(T):
            z_unactivated = z.clone()
            A_z_unactivated = torch.zeros_like(z)

            if self.L > 0:
                A_z_unactivated[:, -self.L:] = self.A * z[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)
            else:
                z_activated = z

            z = A_z_unactivated + z_activated @ self.W.T + x[:, t] @ self.C.T + self.h
            Z.append(z)
            
        # Output is from final state
        output = z @ self.D.T  # shape: [batch_size, N]
        return output, torch.stack(Z)

    def initialize_AWh_random(self):
        A = nn.Parameter(torch.randn(self.L) * 0.01)
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.randn(self.M) * 0.01)
        return A, W, h

    def init_uniform(self, shape):
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

def train_model(model, train_loader, val_loader, test_loader, num_epochs=100, lr=0.001, 
                regularization=True, tau=0.1, M_reg=25, eval_every=50):
    """
    Train the model and evaluate periodically using validation loss for model selection.
    
    Args:
        model (PLRNN): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        regularization (bool): Whether to use regularization
        tau (float): Regularization strength
        M_reg (int): Number of units to regularize
        eval_every (int): Evaluate every N epochs
        
    Returns:
        best_model (PLRNN): Best model based on validation loss
        final_metrics (dict): Final training metrics including test accuracy
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model = None
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'test_accuracy': None  # Will be computed at the end
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = loss_fn(output, y_batch)
            
            if regularization:
                reg_loss = regularization_loss(model, tau, M_reg)
                loss += reg_loss
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output, _ = model(x_batch)
                    val_loss += loss_fn(output, y_batch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            metrics['val_loss'].append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(model)
    
    # Compute final test accuracy using best model
    test_accuracy = test_model(best_model, test_loader)
    metrics['test_accuracy'] = test_accuracy
    print(f"\nFinal test accuracy: {test_accuracy:.2f}%")
    
    return best_model, metrics

def test_model(model, test_loader):
    """
    Evaluate the model on the test set.
    
    Args:
        model (PLRNN): The model to evaluate
        test_loader (DataLoader): Test data loader
        
    Returns:
        accuracy (float): Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output, _ = model(x_batch)  # output: [batch_size, N]
            predictions = torch.argmax(output, dim=1)  # Predicted class indices
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = 100 * correct / total
    return accuracy

def run_memory_task(
    # Task hyperparameters
    T=100,  # Sequence length
    num_train=1000,
    num_val=200,
    num_test=200,
    noise_std=0.01,
    invert_prob=0.5,
    run_id="run_1",
    
    # Model hyperparameters
    M=2,  # Hidden size
    L=1,  # Nonlinear dimensions
    N=2,  # Output dimension (binary classification)
    input_dim=4,  # Input dimension
    
    # Training hyperparameters
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    eval_every=50,
    
    # Regularization hyperparameters
    regularization=True,
    tau=0.1,
    M_reg=25,
):
    """
    Run the contextual memory task experiment with train/val/test split and evaluation.
    """
    # Create datasets and loaders
    print(f"Generating datasets (train: {num_train}, val: {num_val}, test: {num_test} samples)...")
    from load_data import create_datasets, get_data_loaders
    
    train_dataset, val_dataset, test_dataset = create_datasets(
        T=T,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        noise_std=noise_std,
        invert_prob=invert_prob
    )
    
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size
    )
    
    # Initialize model
    print(f"Initializing PLRNN with {M} hidden units ({L} nonlinear)...")
    model = PLRNN(
        M=M,
        L=L,
        N=N,
        input_dim=input_dim
    )
    
    # Train model
    print("Starting training...")
    best_model, final_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        regularization=regularization,
        tau=tau,
        M_reg=M_reg,
        eval_every=eval_every
    )
    
    return best_model, final_metrics 