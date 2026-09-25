import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch import optim
import time
import os
import math
import random
from torch.utils.data import DataLoader, random_split, Dataset
import torchaudio
from torchaudio.transforms import MFCC

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss


class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim, readout_dim, initial_state_fixed_zero=False, dim_x=40):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N
        self.input_dim = input_dim  # Now used for both data and conv in
        self.readout_dim = readout_dim
        self.initial_state_fixed_zero = initial_state_fixed_zero

        self.A, self.W, self.h = self.initialize_AWh_random()
        self.C = nn.Parameter(torch.randn(self.M, self.input_dim) * 0.1)

        self.input_encoder = nn.Sequential(
            nn.Conv1d(dim_x, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, M, kernel_size=3, padding=1)
        )

        self.output_layer = nn.Linear(self.readout_dim, self.N)
        
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
        batch_size, T, dim_x = x.shape
        x = x.transpose(1, 2)  # → [B, 40, T]
        encoded_sequence = self.input_encoder(x)

        T_enc = encoded_sequence.shape[2]
       # encoded_sequence = x[:,:]
        # Initialize hidden state `z`
        z = self.init_uniform((batch_size, self.M))
        latent_states=[]
        for t in range(T_enc):
            latent_states.append(z.detach().clone())
            input_value = encoded_sequence[:,:, t]
            # Keep a copy of z_unactivated before applying ReLU
            z_unactivated = torch.clone(z)
            A_z_unactivated = torch.zeros_like(z_unactivated)

            if self.L > 0:
            
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)

            else:
                z_activated = z
            # Update hidden state using combined input (x[t], m[t]) and matrix C
            z = A_z_unactivated + z_activated @ self.W.t() + input_value @ self.C.t() + self.h
            
        output = self.output_layer(z[:,:self.readout_dim])
        #output=z[:,:self.N]
        latent_states = torch.stack(latent_states, dim=1)
        return output, latent_states


def train_model_audio(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=0.001,
    tau_alpha=0.005,
    M_reg=20,
    regularization=True,
    use_scheduler=True,
    scheduler_type='cosine',
    print_every=1,
    save_best_to=None
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = None

    if use_scheduler:
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
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

            batch_time = time.time() - batch_start
            batch_timer += batch_time
            sample_counter += x_batch.size(0)

            if (batch_idx + 1) % 10 == 0:
                avg_samples_per_sec = sample_counter / batch_timer
                print(f"  [Epoch {epoch+1}] Loss: {loss.item():.4f} | Avg: {avg_samples_per_sec:.1f} samples/s")
                batch_timer = 0.0
                sample_counter = 0

        if scheduler:
            scheduler.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch in val_loader:
                output, _ = model(x_batch)
                val_loss += loss_fn(output, y_batch).item()
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            if save_best_to:
                torch.save({
                    'model_state_dict': best_model_state,
                    'best_epoch': best_epoch,
                    'val_loss': best_val_loss
                }, save_best_to)

        if (epoch + 1) % print_every == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    return best_val_loss, best_model_state



def test_model_audio(model, test_loader):
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



# --- Constants ---
TARGET_WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
LABELS = {word: i for i, word in enumerate(TARGET_WORDS)}

# --- Dataset Class ---
class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, transform=None, allowed_words=TARGET_WORDS):
        self.root_dir = root_dir
        self.transform = transform
        self.allowed_words = allowed_words
        self.samples = []

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path) or label not in self.allowed_words:
                continue
            for fname in os.listdir(label_path):
                if fname.endswith(".wav"):
                    fpath = os.path.join(label_path, fname)
                    self.samples.append((fpath, label))

        print(f"📂 Loaded {len(self.samples)} samples from: {allowed_words}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(fpath)
        features = self.transform(waveform)  # [1, n_mfcc, T]
        features = features.squeeze(0).transpose(0, 1)  # [T, D]
        label_idx = LABELS[label]
        return features, label_idx

# --- Normalize Features ---
class StandardizedDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mfcc, label = self.dataset[idx]
        mfcc = (mfcc - self.mean) / (self.std + 1e-6)
        return mfcc, label

def estimate_global_mfcc_stats(dataset, sample_fraction=0.1, max_samples=10000):
    all_features = []
    indices = random.sample(range(len(dataset)), min(int(len(dataset) * sample_fraction), max_samples))
    for i in indices:
        mfcc, _ = dataset[i]
        all_features.append(mfcc)
    stacked = torch.cat(all_features, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)

# --- Padding for variable-length ---
def collate_pad(batch):
    sequences, labels = zip(*batch)
    dim = sequences[0].shape[1]
    max_len = max(seq.shape[0] for seq in sequences)
    padded = torch.zeros(len(sequences), max_len, dim)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq
    return padded, torch.tensor(labels)

# --- Final Wrapper ---
def get_data_loaders_audio(path_train, path_test, batch_size=128):
    mfcc_transform = MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 40, "n_mels": 64}
    )

    train_set = SpeechCommandsDataset(path_train, transform=mfcc_transform)
    test_set = SpeechCommandsDataset(path_test, transform=mfcc_transform)

    # Normalize features
    mean, std = estimate_global_mfcc_stats(train_set)
    train_set_std = StandardizedDataset(train_set, mean, std)
    test_set_std = StandardizedDataset(test_set, mean, std)

    # Train/val split
    train_size = int(0.9 * len(train_set_std))
    val_size = len(train_set_std) - train_size
    train_subset, val_subset = random_split(train_set_std, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_set_std, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    return train_loader, val_loader, test_loader

