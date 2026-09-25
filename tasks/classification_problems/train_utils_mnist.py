import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import os
import math
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim, readout_dim, initial_state_fixed_zero=False, nonlinearity_type="relu"):
        super(PLRNN, self).__init__()
        self.M = M  # Hidden state dimensionality
        self.L = L  # Number of nonlinear units
        self.N = N  # Output dimensionality (number of classes, 10 for MNIST)
        self.input_dim = input_dim  # Input dimensionality (1 for pixels)
        self.readout_dim = readout_dim
        self.initial_state_fixed_zero = initial_state_fixed_zero
        self.nonlinearity_type = nonlinearity_type
        # Initialize A, W, h
        self.A, self.W, self.h = self.initialize_AWh_random()

        # External input interaction matrix C
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)

        # Readout matrix D for generating classification output
        self.output_layer = nn.Linear(self.readout_dim, 10)
        #self.output_layer = nn.Linear(self.readout_dim, self.N)

        self.input_encoder = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),  # RF=15, wider kernel to capture local strokes
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),  #Pooling                
        nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),  #Pooling               
        nn.Conv1d(32, M, kernel_size=3, stride=1, padding=1)   
        )
        
    def initialize_AWh_random(self):
        # Randomly initialize A, W, h
        A = nn.Parameter(torch.randn(self.L) * 0.01)
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.randn(self.M) * 0.01)
        return A, W, h

    def init_uniform(self, shape):
        # Initialize hidden state uniformly
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def forward(self, x):
        batch_size, T = x.shape  # x has shape [batch_size, T]
        encoded_sequence = self.input_encoder(x.unsqueeze(1))

        T_enc = encoded_sequence.shape[2]
       # encoded_sequence = x[:,:]
       
        # Initialize hidden state `z`
        if self.initial_state_fixed_zero:
          z = torch.zeros(batch_size, self.M, device=x.device)
        else:
          z = self.init_uniform((batch_size, self.M))
        
        # Select nonlinearity function
        if self.nonlinearity_type == "relu":
            nonlin_fn = F.relu
        elif self.nonlinearity_type == "tanh":
            nonlin_fn = torch.tanh
        elif self.nonlinearity_type == "hardtanh":
            nonlin_fn = F.hardtanh
        else:
            raise ValueError(f"Unsupported nonlinearity_type: {self.nonlinearity_type}")

        latent_states=[]
        for t in range(T_enc):
            latent_states.append(z.detach().clone())
            input_value = encoded_sequence[:,:, t]
            # Keep a copy of z_unactivated before applying nonlinearity
            z_unactivated = torch.clone(z)
            A_z_unactivated = torch.zeros_like(z_unactivated)

            if self.L > 0:
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], nonlin_fn(z[:, -self.L:])], dim=1)
            else:
                z_activated = z
            # Update hidden state using combined input (x[t], m[t]) and matrix C
            z = A_z_unactivated + z_activated @ self.W.t() + input_value @ self.C.t() + self.h
        
        output = self.output_layer(z[:,:self.readout_dim])
        #output=z[:,:self.N]
        latent_states = torch.stack(latent_states, dim=1)
        return output, latent_states


class LSTMClassifier(nn.Module):
    def __init__(self, M, L, N, input_dim, readout_dim, initial_state_fixed_zero=False, nonlinearity_type=None):
        super(LSTMClassifier, self).__init__()
        self.M = M  # Hidden state dimensionality (used as LSTM hidden size)
        self.L = L  # Not used for LSTM, kept for interface compatibility
        self.N = N  # Output dimensionality (number of classes, 10 for MNIST)
        self.input_dim = input_dim
        self.readout_dim = readout_dim
        self.initial_state_fixed_zero = initial_state_fixed_zero
        # nonlinearity_type is ignored for LSTM

        self.input_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, M, kernel_size=3, stride=1, padding=1)
        )
        self.lstm = nn.LSTM(input_size=M, hidden_size=M, batch_first=True)
        self.output_layer = nn.Linear(self.readout_dim, 10)

    def forward(self, x):
        batch_size, T = x.shape  # x has shape [batch_size, T]
        encoded_sequence = self.input_encoder(x.unsqueeze(1))  # [batch, M, T_enc]
        encoded_sequence = encoded_sequence.permute(0, 2, 1)  # [batch, T_enc, M]
        T_enc = encoded_sequence.shape[1]

        # Initialize hidden and cell states
        h0 = torch.zeros(1, batch_size, self.M, device=x.device)
        c0 = torch.zeros(1, batch_size, self.M, device=x.device)
        output_seq, (hn, cn) = self.lstm(encoded_sequence, (h0, c0))  # output_seq: [batch, T_enc, M]
        # For classification, use the last hidden state
        output = self.output_layer(output_seq[:, -1, :][:, :self.readout_dim])
        latent_states = output_seq  # [batch, T_enc, M]
        return output, latent_states


class GRUClassifier(nn.Module):
    def __init__(self, M, L, N, input_dim, readout_dim, initial_state_fixed_zero=False, nonlinearity_type=None):
        super(GRUClassifier, self).__init__()
        self.M = M  # Hidden state dimensionality (used as GRU hidden size)
        self.L = L  # Not used for GRU, kept for interface compatibility
        self.N = N  # Output dimensionality (number of classes, 10 for MNIST)
        self.input_dim = input_dim
        self.readout_dim = readout_dim
        self.initial_state_fixed_zero = initial_state_fixed_zero
        # nonlinearity_type is ignored for GRU

        self.input_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, M, kernel_size=3, stride=1, padding=1)
        )
        self.gru = nn.GRU(input_size=M, hidden_size=M, batch_first=True)
        self.output_layer = nn.Linear(self.readout_dim, 10)

    def forward(self, x):
        batch_size, T = x.shape  # x has shape [batch_size, T]
        encoded_sequence = self.input_encoder(x.unsqueeze(1))  # [batch, M, T_enc]
        encoded_sequence = encoded_sequence.permute(0, 2, 1)  # [batch, T_enc, M]
        T_enc = encoded_sequence.shape[1]

        # Initialize hidden state
        h0 = torch.zeros(1, batch_size, self.M, device=x.device)
        output_seq, hn = self.gru(encoded_sequence, h0)  # output_seq: [batch, T_enc, M]
        # For classification, use the last hidden state
        output = self.output_layer(output_seq[:, -1, :][:, :self.readout_dim])
        latent_states = output_seq  # [batch, T_enc, M]
        return output, latent_states



def get_data_loaders(batch_size=64, data_root='mnist_data'):
    transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
    train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

def compute_accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            output, _ = model(x_batch)
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total if total > 0 else 0.0

def train_model(model, train_loader, val_loader, num_epochs=200, lr=0.001,
                tau_alpha=0.005, M_reg=20, regularization=True, save_best_to=None,
                use_scheduler=True, scheduler_type='cosine', print_every=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        correct, total = 0, 0
        epoch_start = time.time()
        batch_timer = 0.0
        sample_counter = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            batch_start = time.time()
            optimizer.zero_grad()
            output, _ = model(x_batch)
            loss = loss_fn(output, y_batch)
            if regularization:
                loss += regularization_loss(model, tau_alpha, M_reg)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            batch_time = time.time() - batch_start
            batch_timer += batch_time
            sample_counter += x_batch.size(0)
            if (batch_idx + 1) % 10 == 0:
                avg_samples_per_sec = sample_counter / batch_timer
                print(f"  [Epoch {epoch+1}| "
                      f"Loss: {loss.item():.4f} | Avg: {avg_samples_per_sec:.1f} samples/s")
                batch_timer = 0.0
                sample_counter = 0
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        if scheduler:
            scheduler.step()
        # Validation loss and accuracy
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_correct, val_total = 0, 0
            for x_batch, y_batch in val_loader:
                output, _ = model(x_batch)
                val_loss += loss_fn(output, y_batch).item()
                _, predicted = torch.max(output, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            if save_best_to:
                # Use the new saving function with model configuration
                save_model_with_config(
                    model=model,
                    save_path=save_best_to,
                    best_epoch=best_epoch,
                    val_loss=best_val_loss,
                    train_loss_history=train_loss_history,
                    val_loss_history=val_loss_history,
                    train_acc_history=train_acc_history,
                    val_acc_history=val_acc_history
                )
        if (epoch + 1) % print_every == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    return best_val_loss, best_model_state, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output, _ = model(x_batch)
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total * 100
    print(f"🧪 Test Accuracy: {accuracy:.2f}%")
    return accuracy


def load_model_from_checkpoint(model_path, device=None):
    """
    Load a trained model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model file (.pt)
        device: Device to load the model on (default: auto-detect)
    
    Returns:
        model: Loaded model instance
        checkpoint_info: Dictionary containing training info (epoch, loss, etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict and other info
    model_state_dict = checkpoint['model_state_dict']
    checkpoint_info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    print(f"Checkpoint info: {checkpoint_info}")
    
    # Determine model type and parameters from the state dict
    # This is a bit hacky but works for the current model structure
    if 'A' in model_state_dict:
        # PLRNN model
        M = model_state_dict['W'].shape[0]
        L = model_state_dict['A'].shape[0]
        
        # Try to get nonlinearity from model_config first, then fallback to checkpoint_info
        model_config = checkpoint_info.get('model_config', {})
        nonlinearity_type = model_config.get('nonlinearity_type', 
                                          checkpoint_info.get('nonlinearity_type', 'relu'))
        
        # If still not found, try to extract from filename
        if nonlinearity_type == 'relu':  # Default value, might be wrong
            import re
            filename = os.path.basename(model_path)
            # Look for nonlinearity type in filename patterns like:
            # "run_L50_tau0.1000_r0_plrnn_tanh_best.pt" -> extract "tanh"
            # "nonlin-tanh" -> extract "tanh"
            nonlinearity_match = re.search(r'plrnn_(\w+)_best|nonlin-(\w+)', filename)
            if nonlinearity_match:
                extracted_nonlin = nonlinearity_match.group(1) or nonlinearity_match.group(2)
                if extracted_nonlin in ['tanh', 'relu', 'gelu', 'hardtanh']:
                    nonlinearity_type = extracted_nonlin
                    print(f"Extracted nonlinearity type from filename: {nonlinearity_type}")
        
        print(f"Detected PLRNN model: M={M}, L={L}, nonlinearity={nonlinearity_type}")
        
        model = PLRNN(
            M=M,
            L=L,
            N=10,  # MNIST has 10 classes
            input_dim=M,
            readout_dim=M,
            nonlinearity_type=nonlinearity_type
        )
    elif 'lstm.weight_ih_l0' in model_state_dict:
        # LSTM model
        M = model_state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
        print(f"Detected LSTM model: M={M}")
        
        model = LSTMClassifier(
            M=M,
            L=0,  # Not used for LSTM
            N=10,
            input_dim=M,
            readout_dim=M
        )
    elif 'gru.weight_ih_l0' in model_state_dict:
        # GRU model
        M = model_state_dict['gru.weight_ih_l0'].shape[0] // 3  # GRU has 3 gates
        print(f"Detected GRU model: M={M}")
        
        model = GRUClassifier(
            M=M,
            L=0,  # Not used for GRU
            N=10,
            input_dim=M,
            readout_dim=M
        )
    else:
        raise ValueError("Could not determine model type from state dict")
    
    # Load the state dict
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model, checkpoint_info


def save_model_with_config(model, save_path, **kwargs):
    """
    Save a model with its configuration and additional metadata.
    
    Args:
        model: The model to save
        save_path: Path where to save the model
        **kwargs: Additional metadata to save with the model
    """
    # Extract model configuration
    config = {}
    if hasattr(model, 'M'):
        config['M'] = model.M
    if hasattr(model, 'L'):
        config['L'] = model.L
    if hasattr(model, 'N'):
        config['N'] = model.N
    if hasattr(model, 'input_dim'):
        config['input_dim'] = model.input_dim
    if hasattr(model, 'readout_dim'):
        config['readout_dim'] = model.readout_dim
    if hasattr(model, 'nonlinearity_type'):
        config['nonlinearity_type'] = model.nonlinearity_type
    if hasattr(model, 'initial_state_fixed_zero'):
        config['initial_state_fixed_zero'] = model.initial_state_fixed_zero
    
    # Determine model type
    if hasattr(model, 'A'):
        config['model_type'] = 'plrnn'
    elif hasattr(model, 'lstm'):
        config['model_type'] = 'lstm'
    elif hasattr(model, 'gru'):
        config['model_type'] = 'gru'
    else:
        config['model_type'] = 'unknown'
    
    # Save everything
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config,
        **kwargs
    }, save_path)
    
    print(f"Model saved to: {save_path}")
    print(f"Model config: {config}")




