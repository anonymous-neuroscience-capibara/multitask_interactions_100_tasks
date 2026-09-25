import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from generate_data import generate_variable_delay_task_dataset_discrete
from rnn_model import PLRNN, LSTM, GRU, train_model_variable_delay, evaluate_model_variable_delay, save_model_with_metrics
from ssm_models import create_ssm_model
import copy
import argparse
import os

def run_variable_delay_task(
    # Task hyperparameters
    seq_len=8,
    static_period=50,
    variable_period=50,
    num_symbols=5,
    num_train=1000,
    num_test=200,
    time_dim=16,
    run_id="run_1",
    # Model hyperparameters
    hidden_size=50,
    nonlin_size=25,
    model_type="plrnn",  # Model type: "plrnn", "lstm", "gru", "s4", "hippo", "mamba"
    nonlinearity_type="relu",  # Nonlinearity: "relu", "tanh", or "gelu" (only for PLRNN)
    
    # Training hyperparameters
    batch_size=64,
    num_epochs=5000,
    learning_rate=0.001,
    eval_every=50,
    include_time_encoding=False,
    results_folder="results/default",
    # Regularization hyperparameters
    tau=0.1,  # Single regularization parameter
):
    """
    Run the variable delay task experiment with train/test split and evaluation.
    
    Task structure:
    1. Static recall period (first static_period time steps)
    2. Variable cue period (next variable_period time steps where cue can occur randomly)
    3. Fixed total sequence length with padding after recall
    """
    # Calculate M_reg as half of hidden_size (rounded to nearest integer)
    M_reg = round(hidden_size / 2)
    
    # Calculate derived parameters
    total_len = static_period + variable_period + seq_len
    if include_time_encoding:
        input_dim = num_symbols + time_dim
    else:
        input_dim = num_symbols
    
    # Create datasets and loaders
    print(f"Generating variable delay datasets (train: {num_train}, test: {num_test} samples)...")
    print(f"Task structure: {static_period} static + {variable_period} variable + {seq_len} recall = {total_len} total steps")
    
    # Generate training data
    x_train, y_train, train_cues = generate_variable_delay_task_dataset_discrete(
        seq_len=seq_len,
        num_symbols=num_symbols,
        num_samples=num_train,
        static_period=static_period,
        variable_period=variable_period,
        include_time_encoding=include_time_encoding,
        time_dim=time_dim
    )
    
    # Generate test data
    x_test, y_test, test_cues = generate_variable_delay_task_dataset_discrete(
        seq_len=seq_len,
        num_symbols=num_symbols,
        num_samples=num_test,
        static_period=static_period,
        variable_period=variable_period,
        include_time_encoding=include_time_encoding,
        time_dim=time_dim
    )
    
    print(f"Data shapes:")
    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  x_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"  Cue positions - Train: {min(train_cues)}-{max(train_cues)}, Test: {min(test_cues)}-{max(test_cues)}")
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initialize model based on model_type
    if model_type.lower() == "lstm":
        print(f"Initializing LSTM with {hidden_size} hidden units...")
        model = LSTM(
            M=hidden_size,
            N=num_symbols,
            input_dim=input_dim
        )
    elif model_type.lower() == "gru":
        print(f"Initializing GRU with {hidden_size} hidden units...")
        model = GRU(
            M=hidden_size,
            N=num_symbols,
            input_dim=input_dim
        )
    elif model_type.lower() == "plrnn":
        print(f"Initializing PLRNN with {hidden_size} hidden units ({nonlin_size} nonlinear, nonlinearity: {nonlinearity_type})...")
        model = PLRNN(
            M=hidden_size,
            L=nonlin_size,
            N=num_symbols,
            input_dim=input_dim,
            nonlinearity_type=nonlinearity_type
        )
    elif model_type.lower() in ["s4", "hippo", "mamba"]:
        print(f"Initializing {model_type.upper()} with {hidden_size} hidden units...")
        model = create_ssm_model(
            model_type=model_type,
            M=hidden_size,
            N=num_symbols,
            input_dim=input_dim
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'plrnn', 'lstm', 'gru', 's4', 'hippo', or 'mamba'.")
    
    # Train model
    print("Starting training...")
    best_model, final_metrics = train_model_variable_delay(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        seq_len=seq_len,
        cue_positions_train=train_cues,
        cue_positions_test=test_cues,
        num_epochs=num_epochs,
        eval_every=eval_every,
        lr=learning_rate,
        tau=tau,
        M_reg=M_reg
    )
    
    return best_model, final_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Variable Delay Task Training')
    
    # Task hyperparameters
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--static_period', type=int, required=True)
    parser.add_argument('--variable_period', type=int, required=True)
    parser.add_argument('--num_symbols', type=int, required=True)
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=200)
    parser.add_argument('--time_dim', type=int, default=16)
    
    # Model hyperparameters
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--nonlin_size', type=int, default=25)  # Made optional, will be ignored for LSTM/GRU/SSMs
    parser.add_argument('--model_type', type=str, default='plrnn', choices=['plrnn', 'lstm', 'gru', 's4', 'hippo', 'mamba'])
    parser.add_argument('--nonlinearity_type', type=str, default='relu', choices=['relu', 'tanh', 'gelu'])
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--tau', type=float, default=0.1)  # Made optional, will be ignored for LSTM/GRU/SSMs
    parser.add_argument('--include_time_encoding', type=bool, default=False)
    
    # Results organization
    parser.add_argument('--results_folder', type=str, default='results/default')
    
    # Run identification
    parser.add_argument('--run_id', type=str, required=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create hyperparameters dictionary from args
    hyperparams = vars(args)

    # --- Construct the save directory based on hyperparameters ---
    # Create a clear, organized folder structure
    if args.model_type.lower() == "plrnn":
        # For PLRNN: include model type, nonlinearity, L, tau, and task parameters
        config_folder_name = (
            f"plrnn_{args.nonlinearity_type}_L{args.nonlin_size}_tau{args.tau:.3f}_"
            f"seq{args.seq_len}_static{args.static_period}_var{args.variable_period}_sym{args.num_symbols}"
        )
    elif args.model_type.lower() in ["s4", "hippo", "mamba"]:
        # For SSM models: include model type and task parameters (L and tau don't apply)
        config_folder_name = (
            f"{args.model_type}_seq{args.seq_len}_static{args.static_period}_var{args.variable_period}_sym{args.num_symbols}"
        )
    else:
        # For LSTM and GRU: include model type and task parameters (L and tau don't apply)
        config_folder_name = (
            f"{args.model_type}_seq{args.seq_len}_static{args.static_period}_var{args.variable_period}_sym{args.num_symbols}"
        )
    save_directory = os.path.join(args.results_folder, config_folder_name)

    # Run experiment, passing all args as keyword arguments
    model, final_metrics = run_variable_delay_task(**hyperparams)

    if model is None:
         print(f"Training failed or stopped very early for run {args.run_id}. No model saved.")
    else:
        # Save model and metrics using the constructed directory
        # Pass the full hyperparams dict so run_id can be used for the filename
        model_path, meta_path = save_model_with_metrics(
            model=model,
            metrics=final_metrics,
            hyperparams=hyperparams, # Pass the full dict including run_id
            save_dir=save_directory # Pass the config-specific directory
        )
        print("\nFinal Test Results (Best Model):")
        print(f"Symbol Accuracy: {final_metrics.get('symbol_accuracy', 'N/A'):.2f}%")
        print(f"Sequence Accuracy: {final_metrics.get('sequence_accuracy', 'N/A'):.2f}%") 