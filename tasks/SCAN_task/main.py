import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import load_scan_data, SCANDataset, build_vocabs, collate_fn
from PLRNN_model import PLRNN, DecoderALRNN, train_model, save_model_and_metrics
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train SCAN task model')
    parser.add_argument('--L_encoder', type=int, required=True, help='Nonlinear dimensions for encoder')
    parser.add_argument('--L_decoder', type=int, required=True, help='Nonlinear dimensions for decoder')
    parser.add_argument('--M', type=int, default=128, help='Hidden size')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Input embedding dimension')
    parser.add_argument('--output_embedding_dim', type=int, default=64, help='Output embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=75, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--regularization', type=bool, default=True, help='Whether to use regularization')
    parser.add_argument('--tau', type=float, default=0.1, help='Regularization strength')
    parser.add_argument('--M_reg', type=int, default=50, help='Number of units for regularization')
    parser.add_argument('--initial_teacher_forcing', type=float, default=0.0, help='Initial teacher forcing ratio')
    parser.add_argument('--end_token_weight', type=float, default=5.0, help='Weight for end token in loss')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model and metrics')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data
    train_inputs, train_outputs = load_scan_data('Data/tasks_train_simple.txt')
    test_inputs, test_outputs = load_scan_data('Data/tasks_test_simple.txt')

    # Build vocabularies
    input_vocab, output_vocab = build_vocabs(train_inputs, train_outputs)

    # Create datasets
    train_dataset = SCANDataset(train_inputs, train_outputs, input_vocab, output_vocab)
    test_dataset = SCANDataset(test_inputs, test_outputs, input_vocab, output_vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input embedding (for commands)
    input_embedding = nn.Embedding(len(input_vocab), args.embedding_dim)

    # Output embedding (for actions)
    output_embedding = nn.Embedding(len(output_vocab), args.output_embedding_dim)

    # Encoder AL-RNN
    encoder = PLRNN(M=args.M, L=args.L_encoder, N=args.M, input_dim=args.embedding_dim)

    # Decoder AL-RNN
    decoder = DecoderALRNN(M=args.M, L=args.L_decoder, output_vocab_size=len(output_vocab), 
                          output_embed_dim=args.output_embedding_dim, output_vocab=output_vocab)

    # Train model
    token_accuracy, sequence_accuracy = train_model(
        encoder, decoder, input_embedding, output_embedding, 
        train_loader, test_loader,
        num_epochs=args.num_epochs, lr=args.learning_rate, 
        regularization=args.regularization, tau=args.tau, M_reg=args.M_reg,
        initial_teacher_forcing=args.initial_teacher_forcing, end_token_weight=args.end_token_weight,
        output_vocab=output_vocab, device=device
    )

    # Save model and metrics
    hyperparams = vars(args)
    save_model_and_metrics(
        encoder, decoder, input_embedding, output_embedding,
        token_accuracy, sequence_accuracy,
        hyperparams, args.save_dir
    )

if __name__ == "__main__":
    main()