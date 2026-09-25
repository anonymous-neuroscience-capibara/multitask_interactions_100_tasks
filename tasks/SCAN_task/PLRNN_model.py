import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import os
import json
import sys
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
        x: [batch_size, T_in, input_dim]
        Returns:
        - final latent state z [batch_size, M]
        """
        batch_size, T, _ = x.shape
        z = torch.zeros(batch_size, self.M, device=x.device)
    
        for t in range(T):
            z_unactivated = z.clone()
            A_z_unactivated = torch.zeros_like(z)
    
            if self.L > 0:
                A_z_unactivated[:, -self.L:] = self.A * z[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)
            else:
                z_activated = z
    
            z = A_z_unactivated + z_activated @ self.W.T + x[:, t] @ self.C.T + self.h
    
        return z

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

class DecoderALRNN(nn.Module):
    def __init__(self, M, L, output_vocab_size, output_embed_dim, output_vocab):
        super().__init__()
        self.M = M
        self.L = L
        self.output_vocab_size = output_vocab_size
        self.output_embed_dim = output_embed_dim
        self.output_vocab = output_vocab
        
        self.embedding = nn.Embedding(output_vocab_size, output_embed_dim)
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.C = nn.Parameter(torch.randn(self.M, output_embed_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(output_vocab_size, self.M) * 0.1)

    def initialize_AWh_random(self):
        A = nn.Parameter(torch.randn(self.L) * 0.01)
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.randn(self.M) * 0.01)
        return A, W, h

    def forward(self, target_outputs=None, init_state=None, teacher_forcing_ratio=1.0, max_len=None):
        batch_size = init_state.size(0)
        device = init_state.device
    
        z = init_state
        outputs = []
    
        # decide mode:
        free_running = (target_outputs is None) or (teacher_forcing_ratio == 0.0)
    
        if free_running:
            max_steps = max_len if max_len is not None else 50
        else:
            max_steps = target_outputs.size(1)
    
        inputs = torch.full((batch_size,), self.output_vocab.token_to_idx['<start>'], device=device, dtype=torch.long)
    
        for t in range(max_steps):
            x_embedded = self.embedding(inputs)
    
            # AL-RNN update
            z_unactivated = z.clone()
            A_z_unactivated = torch.zeros_like(z)
    
            if self.L > 0:
                A_z_unactivated[:, -self.L:] = self.A * z[:, -self.L:]
                z_activated = torch.cat([z[:, :-self.L], F.relu(z[:, -self.L:])], dim=1)
            else:
                z_activated = z
    
            z = A_z_unactivated + z_activated @ self.W.T + x_embedded @ self.C.T + self.h
    
            logits = z @ self.D.T  # [batch_size, vocab_size]
            outputs.append(logits.unsqueeze(1))
    
            # Sample next input
            if (not free_running) and (torch.rand(1).item() < teacher_forcing_ratio):
                inputs = target_outputs[:, t]
            else:
                inputs = torch.argmax(logits, dim=-1)
    
        outputs = torch.cat(outputs, dim=1)
        return outputs, None


def generate_sequence(encoder, decoder, input_embedding, output_vocab, input_sentence, 
                       max_len=30, device="cpu"):
    encoder.eval()
    decoder.eval()

    input_tokens = input_sentence.strip().split()
    input_ids = input_vocab.encode(input_tokens)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T_in]

    with torch.no_grad():
        embedded_input = input_embedding(input_tensor)
        final_latent = encoder(embedded_input)

        # Start decoding
        start_token_id = self.output_vocab.token_to_idx['<start>']
        end_token_id = self.output_vocab.token_to_idx['<end>']

        inputs = torch.tensor([[start_token_id]], dtype=torch.long, device=device)  # [1, 1]
        generated_ids = []

        z = final_latent  # initialize latent state
        for _ in range(max_len):
            x_embedded = decoder.embedding(inputs).squeeze(1)  # [1, output_embed_dim]

            # AL-RNN update
            z_unactivated = z.clone()
            A_z_unactivated = torch.zeros_like(z)
            if decoder.L > 0:
                A_z_unactivated[:, -decoder.L:] = decoder.A * z[:, -decoder.L:]
                z_activated = torch.cat([z[:, :-decoder.L], F.relu(z[:, -decoder.L:])], dim=1)
            else:
                z_activated = z

            z = A_z_unactivated + z_activated @ decoder.W.T + x_embedded @ decoder.C.T + decoder.h

            logits = z @ decoder.D.T  # [1, output_vocab_size]
            next_token = torch.argmax(logits, dim=-1)  # [1]
            token_id = next_token.item()

            if token_id == end_token_id:
                break

            generated_ids.append(token_id)

            inputs = next_token.unsqueeze(0)  # [1, 1]

    # Decode to tokens
    generated_tokens = output_vocab.decode(generated_ids, remove_special_tokens=True)

    return generated_tokens


def train_model(encoder, decoder, input_embedding, output_embedding, 
                train_loader, test_loader, 
                num_epochs=100, lr=0.001, 
                regularization=True, tau=0.1, M_reg=25, initial_teacher_forcing=1.0, end_token_weight=5., 
                output_vocab=None, device="cpu"):

    encoder.to(device)
    decoder.to(device)
    input_embedding.to(device)
    output_embedding.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + 
        list(input_embedding.parameters()) + list(output_embedding.parameters()), 
        lr=lr
    )

    # Define token-specific weights
    vocab_size = len(output_vocab)
    weights = torch.ones(vocab_size)
    
    end_token_idx = output_vocab.token_to_idx['<end>']
    weights[end_token_idx] = end_token_weight
    
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=output_vocab.token_to_idx['<pad>'])

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        # Scheduled teacher forcing: linearly decay to 0.05
        teacher_forcing_ratio = max(initial_teacher_forcing * (1 - epoch / num_epochs), 0.0)

        for x_batch, y_batch, _, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            embedded_input = input_embedding(x_batch)
            final_latent = encoder(embedded_input)

            # If teacher forcing ratio > 0, use it. Otherwise pure free-running.
            if teacher_forcing_ratio > 0:
                logits, _ = decoder(target_outputs=y_batch, init_state=final_latent, teacher_forcing_ratio=teacher_forcing_ratio)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y_batch.view(-1)

                mask = (targets_flat != output_vocab.token_to_idx["<pad>"])
                loss = loss_fn(logits_flat, targets_flat)
            else:
                # Free-running decoding
                max_len = y_batch.size(1)
                logits, _ = decoder(target_outputs=None, init_state=final_latent, teacher_forcing_ratio=0.0, max_len=max_len)

                batch_loss = 0.0
                batch_tokens = 0

                for i in range(x_batch.size(0)):
                    pred_logits = logits[i]
                    true_seq = y_batch[i]
                    true_seq = true_seq[(true_seq != output_vocab.token_to_idx['<pad>']) &
                                        (true_seq != output_vocab.token_to_idx['<start>']) &
                                        (true_seq != output_vocab.token_to_idx['<end>'])]
                    true_seq = true_seq.tolist()

                    n = len(true_seq)
                    if n > 0:
                        pred_logits = pred_logits[:n]
                        true_targets = torch.tensor(true_seq, device=device)
                        l = loss_fn(pred_logits, true_targets)
                        batch_loss += l * n
                        batch_tokens += n

                loss = batch_loss / batch_tokens

            # Optional regularization
            if regularization:
                loss+=regularization_loss(encoder, tau, M_reg)
                loss+=regularization_loss(decoder, tau, M_reg)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Teacher Forcing: {teacher_forcing_ratio:.2f}")

        if (epoch + 1) % 10 == 0:
            token_accuracy, sequence_accuracy = test_model_token_and_sequence_level(encoder, decoder, input_embedding, output_vocab, test_loader, device=device)
            print(f"Epoch [{epoch+1}/{num_epochs}] Token Accuracy: {token_accuracy:.2f}% | Sequence Accuracy: {sequence_accuracy:.2f}%")

    # Get final test accuracy
    token_accuracy, sequence_accuracy = test_model_token_and_sequence_level(encoder, decoder, input_embedding, output_vocab, test_loader, device=device)
    return token_accuracy, sequence_accuracy

def test_model_token_and_sequence_level(encoder, decoder, input_embedding, output_vocab, test_loader, device="cpu", max_len=50):
    encoder.eval()
    decoder.eval()
    
    start_token_id = output_vocab.token_to_idx['<start>']
    end_token_id = output_vocab.token_to_idx['<end>']
    pad_token_id = output_vocab.token_to_idx['<pad>']

    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for x_batch, y_batch, input_lens, output_lens in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size = x_batch.size(0)

            embedded_input = input_embedding(x_batch)
            final_latent = encoder(embedded_input)

            # Initialize
            inputs = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
            z = final_latent
            generated_ids = [[] for _ in range(batch_size)]
            finished = [False for _ in range(batch_size)]

            for _ in range(max_len):
                x_embedded = decoder.embedding(inputs).squeeze(1)
                z_unactivated = z.clone()
                A_z_unactivated = torch.zeros_like(z)

                if decoder.L > 0:
                    A_z_unactivated[:, -decoder.L:] = decoder.A * z[:, -decoder.L:]
                    z_activated = torch.cat([z[:, :-decoder.L], F.relu(z[:, -decoder.L:])], dim=1)
                else:
                    z_activated = z

                z = A_z_unactivated + z_activated @ decoder.W.T + x_embedded @ decoder.C.T + decoder.h

                logits = z @ decoder.D.T
                preds = torch.argmax(logits, dim=-1)

                next_inputs = []
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = preds[i].item()
                        generated_ids[i].append(token_id)
                        if token_id == end_token_id:
                            finished[i] = True
                        next_inputs.append(token_id)
                    else:
                        next_inputs.append(pad_token_id)

                inputs = torch.tensor(next_inputs, device=device).unsqueeze(1)

            # Compare predictions vs targets
            for i in range(batch_size):
                pred_seq = generated_ids[i]
                if end_token_id in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(end_token_id)]

                true_seq = y_batch[i]
                true_seq = true_seq[(true_seq != pad_token_id) &
                                    (true_seq != start_token_id) &
                                    (true_seq != end_token_id)]
                true_seq = true_seq.tolist()

                n = len(true_seq)
                pred_seq = pred_seq[:n]

                correct_tokens += sum([p == t for p, t in zip(pred_seq, true_seq)])
                total_tokens += n

                if pred_seq == true_seq:
                    correct_sequences += 1
                total_sequences += 1

    token_accuracy = 100 * correct_tokens / total_tokens
    sequence_accuracy = 100 * correct_sequences / total_sequences
    print(f"Token-level Test Accuracy (without teacher forcing): {token_accuracy:.2f}%")
    print(f"Sequence-level Test Accuracy (without teacher forcing): {sequence_accuracy:.2f}%")

    return token_accuracy, sequence_accuracy

def save_model_and_metrics(encoder, decoder, input_embedding, output_embedding, 
                         token_accuracy, sequence_accuracy, 
                         hyperparams, save_dir):
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'input_embedding_state_dict': input_embedding.state_dict(),
        'output_embedding_state_dict': output_embedding.state_dict(),
    }, os.path.join(save_dir, 'model.pt'))
    
    # Save metrics and hyperparameters
    metrics = {
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'hyperparameters': hyperparams
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)