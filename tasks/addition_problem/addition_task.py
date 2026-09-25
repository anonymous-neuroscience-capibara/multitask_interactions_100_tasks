import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from generate_data import create_datasets
from rnn_model import PLRNN, train_model, evaluate_model, save_model_with_metrics
import copy
import argparse
import os

def run_addition_task(
    # Task hyperparameters
    T=50,  # Sequence length
    num_train=2000,
    num_test=400,
    run_id="run_1",
    
    # Model hyperparameters
    M=50,  # Hidden size
    L=25,  # Nonlinear dimensions
    N=1,   # Output dimension (sum)
    input_dim=2,  # Input dimension (value + mask)
    
    # Training hyperparameters
    batch_size=100,
    num_epochs=5000,
    learning_rate=0.001,
    eval_every=100,
    
    # Regularization hyperparameters
    regularization=True,
    tau=1.0,
    M_reg=25,
):
    """
    Run the addition task experiment with train/test split and evaluation.
    """
    # Create datasets and loaders
    print(f"Generating datasets (train: {num_train}, test: {num_test} samples)...")
    
    # Generate training and test datasets
    train_dataset, test_dataset = create_datasets(
        seq_len=T,
        N_total=num_train + num_test,
        num_train=num_train,
        num_test=num_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
        test_loader=test_loader,
        T=T,
        num_epochs=num_epochs,
        eval_every=eval_every,
        lr=learning_rate,
        regularization=regularization,
        tau=tau,
        M_reg=M_reg
    )
    
    return best_model, final_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Addition Task Training')
    
    # Task hyperparameters
    parser.add_argument('--T', type=int, required=True)
    parser.add_argument('--num_train', type=int, default=2000)
    parser.add_argument('--num_test', type=int, default=400)
    
    # Model hyperparameters
    parser.add_argument('--M', type=int, default=50)
    parser.add_argument('--L', type=int, required=True)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--input_dim', type=int, default=2)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--eval_every', type=int, default=100)
    
    # Regularization parameters
    parser.add_argument('--regularization', type=bool, default=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--M_reg', type=int, default=25)
    
    # Run identification
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create training hyperparameters dictionary (excluding save_dir)
    train_params = {k: v for k, v in vars(args).items() if k != 'save_dir'}

    # Run experiment with training parameters only
    model, final_metrics = run_addition_task(**train_params)

    if model is None:
         print(f"Training failed or stopped very early for run {args.run_id}. No model saved.")
    else:
        # Save model and metrics using the provided save directory
        model_path, meta_path = save_model_with_metrics(
            model=model,
            metrics=final_metrics,
            hyperparams=vars(args),  # Pass all args for saving
            save_dir=args.save_dir
        )

        print("\nFinal Test Results:")
        print(f"Train Loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Test Loss: {final_metrics.get('test_loss', 'N/A'):.4f}")
