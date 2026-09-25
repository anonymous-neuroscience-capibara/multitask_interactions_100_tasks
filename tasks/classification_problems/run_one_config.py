import argparse
import os
import torch
import json

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.run)

    if args.dataset == "smnist":
        from train_utils_mnist import train_model, test_model, PLRNN, LSTMClassifier, GRUClassifier, get_data_loaders
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, data_root=args.mnist_data_root)
        input_dim = args.M
        N = 10
        model_kwargs = dict(readout_dim=args.M, initial_state_fixed_zero=args.fixed_init, nonlinearity_type=args.nonlinearity_type)
        if getattr(args, 'rnn_type', 'plrnn') == 'lstm':
            model = LSTMClassifier(
                M=args.M,
                L=args.L,  # Not used for LSTM, but kept for interface
                N=N,
                input_dim=input_dim,
                readout_dim=args.M,
                initial_state_fixed_zero=args.fixed_init
            )
            regularization = False
        elif getattr(args, 'rnn_type', 'plrnn') == 'gru':
            model = GRUClassifier(
                M=args.M,
                L=args.L,  # Not used for GRU, but kept for interface
                N=N,
                input_dim=input_dim,
                readout_dim=args.M,
                initial_state_fixed_zero=args.fixed_init
            )
            regularization = False
        else:
            model = PLRNN(
                M=args.M,
                L=args.L,
                N=N,
                input_dim=input_dim,
                **model_kwargs
            )
            regularization = True
        # Update results directory and filenames to include rnn_type and nonlinearity_type
        results_folder = getattr(args, 'results_folder', None)
        if results_folder is None:
            results_folder = f"results_{args.dataset}_{args.rnn_type}"
        config_id = f"L{args.L}_tau{args.tau:.4f}_r{args.run}_{args.rnn_type}_{args.nonlinearity_type}"
        config_dir = os.path.join(results_folder, f"L_{args.L}_tau_{args.tau}")
        os.makedirs(config_dir, exist_ok=True)
        model_path = os.path.join(config_dir, f"run_{config_id}_best.pt")

    elif args.dataset == "speech":
        from train_utils_audio import train_model_audio as train_model
        from train_utils_audio import test_model_audio as test_model
        from train_utils_audio import get_data_loaders_audio
        from train_utils_audio import PLRNN
        train_loader, val_loader, test_loader = get_data_loaders_audio(
            args.audio_train_path, args.audio_test_path, batch_size=args.batch_size
        )
        input_dim = args.M
        N = 10
        model_kwargs = dict(readout_dim=args.M, initial_state_fixed_zero=args.fixed_init, dim_x=40, nonlinearity_type=args.nonlinearity_type)

    elif args.dataset == "imdb":
        from train_utils_imdb import train_model_text as train_model
        from train_utils_imdb import test_model_text as test_model
        from train_utils_imdb import get_data_loaders_imdb, PLRNN
        train_loader, val_loader, test_loader, embedding_matrix = get_data_loaders_imdb(args.glove_path, batch_size=args.batch_size)
        input_dim = 100
        N = 2
        model_kwargs = dict(readout_dim=args.M, initial_state_fixed_zero=args.fixed_init, nonlinearity_type=args.nonlinearity_type)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.dataset == "smnist":
        val_loss, _, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            tau_alpha=args.tau,
            M_reg=50,
            save_best_to=model_path,
            regularization=regularization
        )
        # Compute test loss and accuracy
        model.eval()
        test_loss = 0
        test_correct, test_total = 0, 0
        loss_fn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output, _ = model(x_batch)
                test_loss += loss_fn(output, y_batch).item()
                _, predicted = torch.max(output, 1)
                test_total += y_batch.size(0)
                test_correct += (predicted == y_batch).sum().item()
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        summary = {
            "run": args.run,
            "L": args.L,
            "tau": args.tau,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "fixed_init": args.fixed_init,
            "nonlinearity_type": args.nonlinearity_type,
            "rnn_type": args.rnn_type,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history
        }
    elif args.dataset == "imdb":
        val_loss, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            embedding_matrix=embedding_matrix,
            num_epochs=args.epochs,
            lr=args.lr,
            tau_alpha=args.tau,
            M_reg=50,
            save_best_to=model_path
        )
        acc = test_model(model, test_loader, embedding_matrix)
        summary = {
            "run": args.run,
            "L": args.L,
            "tau": args.tau,
            "val_loss": val_loss,
            "test_acc": acc,
            "fixed_init": args.fixed_init,
            "nonlinearity_type": args.nonlinearity_type,
            "rnn_type": args.rnn_type
        }
    else:
        val_loss, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            tau_alpha=args.tau,
            M_reg=50,
            save_best_to=model_path,
            regularization=regularization
        )
        acc = test_model(model, test_loader)
        summary = {
            "run": args.run,
            "L": args.L,
            "tau": args.tau,
            "val_loss": val_loss,
            "test_acc": acc,
            "fixed_init": args.fixed_init,
            "nonlinearity_type": args.nonlinearity_type,
            "rnn_type": args.rnn_type
        }

    json_path = os.path.join(config_dir, f"run_{config_id}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["smnist", "speech", "imdb"], required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fixed_init", action="store_true")
    parser.add_argument("--audio_train_path", type=str, default=None)
    parser.add_argument("--audio_test_path", type=str, default=None)
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt")
    parser.add_argument("--nonlinearity_type", type=str, default="relu", choices=["relu", "tanh", "hardtanh"], help="Type of nonlinearity to use in PLRNN (relu, tanh, hardtanh)")
    parser.add_argument("--mnist_data_root", type=str, default="mnist_data", help="Path to MNIST data directory")
    parser.add_argument("--rnn_type", type=str, default="plrnn", choices=["plrnn", "lstm", "gru"], help="Type of RNN cell to use (plrnn, lstm, gru)")
    parser.add_argument("--results_folder", type=str, default=None, help="Base folder for saving results (default: results_<dataset>_<rnn_type>)")

    args = parser.parse_args()
    main(args)
