import os
import re
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset, DownloadMode
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MAR import regularization_loss

# ---------------------- MODEL ----------------------
class PLRNN(nn.Module):
    def __init__(self, M, L, N, input_dim, readout_dim, initial_state_fixed_zero=False):
        super(PLRNN, self).__init__()
        self.M = M
        self.L = L
        self.N = N
        self.input_dim = input_dim
        self.readout_dim = readout_dim
        self.initial_state_fixed_zero = initial_state_fixed_zero

        self.A, self.W, self.h = self.initialize_AWh_random()
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, M)
        )
        self.output_layer = nn.Linear(readout_dim, N)

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

    def forward(self, x):
        batch_size, T, _ = x.shape
        z = self.init_uniform((batch_size, self.M))
        latent_states = []

        for t in range(T):
            x_t = x[:, t, :]
            u_t = self.input_encoder(x_t)

            z_unactivated = z.clone()
            A_z_unactivated = torch.zeros_like(z_unactivated)
            if self.L > 0:
                A_z_unactivated[:, -self.L:] = self.A * z_unactivated[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)
            else:
                z_activated = z

            z = A_z_unactivated + z_activated @ self.W.t() + u_t + self.h
            latent_states.append(z.detach().clone())

        output = self.output_layer(z[:, :self.readout_dim])
        latent_states = torch.stack(latent_states, dim=1)
        return output, latent_states

# ---------------------- TRAINING ----------------------
def train_model_text(
    model,
    train_loader,
    val_loader,
    embedding_matrix,
    num_epochs=10,
    lr=1e-3,
    tau_alpha=0.01,
    M_reg=20,
    regularization=True,
    save_best_to=None
):
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    glove_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            embedded = glove_embedding(input_ids)

            optimizer.zero_grad()
            outputs, _ = model(embedded)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loss = criterion(outputs, labels)

            if regularization:
                loss += regularization_loss(model, tau_alpha, M_reg)

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                embedded = glove_embedding(input_ids)
                outputs, _ = model(embedded)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        acc = correct / total
        print(f"Train Loss: {total_loss:.4f} | Accuracy: {acc:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            if save_best_to:
                torch.save({
                    'model_state_dict': best_model_state,
                    'val_loss': best_val_loss
                }, save_best_to)

    return best_val_loss, best_model_state

# ---------------------- TESTING ----------------------
def test_model_text(model, test_loader, embedding_matrix):
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.eval()
    model.to(device)
    glove_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            embedded = glove_embedding(input_ids)
            output, _ = model(embedded)
            pred = torch.argmax(output, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"🧪 Test Accuracy: {acc:.2f}%")
    return acc

# ---------------------- TOKENIZATION ----------------------
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# ---------------------- DATASET ----------------------
class IMDbGloVeDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=256):
        self.labels = labels
        self.sequences = []

        for tokens in texts:
            idxs = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
            if len(idxs) < max_len:
                idxs += [word2idx["<PAD>"]] * (max_len - len(idxs))
            else:
                idxs = idxs[:max_len]
            self.sequences.append(torch.tensor(idxs, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx])

# ---------------------- GLOVE HANDLING ----------------------
def load_glove_embeddings(glove_path):
    glove_dict = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_dict[word] = vector
    return glove_dict

def build_vocab_and_embeddings(tokenized_texts, glove_dict, vocab_size=20000, embedding_dim=100):
    all_words = [word for tokens in tokenized_texts for word in tokens]
    vocab_counter = Counter(all_words)
    vocab = ["<PAD>", "<UNK>"] + [word for word, _ in vocab_counter.most_common(vocab_size)]
    word2idx = {word: i for i, word in enumerate(vocab)}

    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for i, word in enumerate(vocab):
        embedding_matrix[i] = glove_dict.get(word, np.random.normal(scale=0.6, size=(embedding_dim,)))
    return word2idx, torch.tensor(embedding_matrix, dtype=torch.float32)

# ---------------------- DATALOADER WRAPPER ----------------------
def get_data_loaders_imdb(glove_path, batch_size=64, max_len=128):
 
    
    dataset = load_dataset("imdb", download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS)
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    tokenized_train = [simple_tokenizer(t) for t in train_texts]
    tokenized_test = [simple_tokenizer(t) for t in test_texts]

    glove_dict = load_glove_embeddings(glove_path)
    word2idx, embedding_matrix = build_vocab_and_embeddings(tokenized_train, glove_dict)

    train_dataset = IMDbGloVeDataset(tokenized_train, train_labels, word2idx, max_len=max_len)
    test_dataset = IMDbGloVeDataset(tokenized_test, test_labels, word2idx, max_len=max_len)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, embedding_matrix
