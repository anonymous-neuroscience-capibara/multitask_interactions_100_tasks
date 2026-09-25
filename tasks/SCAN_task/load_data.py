def load_scan_data(filepath):
    inputs, outputs = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if 'IN:' in line and 'OUT:' in line:
                in_part = line.split('IN:')[1].split('OUT:')[0].strip()
                out_part = line.split('OUT:')[1].strip()
                inputs.append(in_part)
                outputs.append(out_part)
    return inputs, outputs

# Usage:
train_inputs, train_outputs = load_scan_data('Data/tasks_train_simple.txt')
test_inputs, test_outputs = load_scan_data('Data/tasks_test_simple.txt')


from torch.utils.data import Dataset, DataLoader
import torch

class SCANDataset(Dataset):
    def __init__(self, inputs, outputs, input_vocab, output_vocab):
        self.inputs = inputs
        self.outputs = outputs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tokens = self.inputs[idx].strip().split()
        output_tokens = self.outputs[idx].strip().split()
    
        input_ids = self.input_vocab.encode(input_tokens)  # input: no special tokens
        output_ids = self.output_vocab.encode(output_tokens)  # output: manual below
    
        # Only manually add <end> token
        output_ids.append(self.output_vocab.token_to_idx["<end>"])
    
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(output_ids, dtype=torch.long)


def collate_fn(batch):
    input_seqs, output_seqs = zip(*batch)
    input_lens = [len(seq) for seq in input_seqs]
    output_lens = [len(seq) for seq in output_seqs]

    input_seqs_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    output_seqs_padded = torch.nn.utils.rnn.pad_sequence(output_seqs, batch_first=True, padding_value=0)

    return input_seqs_padded, output_seqs_padded, input_lens, output_lens


class Vocab:
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.counter = 0
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.start_token = "<start>"
        self.end_token = "<end>"

        # Initialize special tokens
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)
        self.add_token(self.start_token)
        self.add_token(self.end_token)

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.counter
            self.idx_to_token[self.counter] = token
            self.counter += 1

    def encode(self, tokens, add_special_tokens=False):
        if add_special_tokens:
            tokens = [self.start_token] + tokens + [self.end_token]
        return [self.token_to_idx.get(token, self.token_to_idx[self.unk_token]) for token in tokens]

    def decode(self, indices, remove_special_tokens=False):
        tokens = [self.idx_to_token.get(idx, self.unk_token) for idx in indices]
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in (self.start_token, self.end_token, self.pad_token)]
        return tokens

    def __len__(self):
        return len(self.token_to_idx)

def build_vocabs(train_inputs, train_outputs):
    input_vocab = Vocab()
    output_vocab = Vocab()

    for sentence in train_inputs:
        for token in sentence.strip().split():
            input_vocab.add_token(token)

    for sentence in train_outputs:
        for token in sentence.strip().split():
            output_vocab.add_token(token)

    return input_vocab, output_vocab
# Usage:
input_vocab, output_vocab = build_vocabs(train_inputs, train_outputs)