import os
import subprocess
import argparse
import json
import torch
from memory_task import run_memory_task

def set_thread_limits():
    """Set thread limits for PyTorch."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run contextual memory task experiments')
    
    # Task hyperparameters
    parser.add_argument('--T', type=int, default=100, help='Sequence length')
    parser.add_argument('--num_train', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=200, help='Number of test samples')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Standard deviation of noise on data')
    parser.add_argument('--invert_prob', type=float, default=0.5, help='Probability of inverting context')
    
    # Model hyperparameters
    parser.add_argument('--M', type=int, default=10, help='Hidden size')
    parser.add_argument('--L', type=int, default=5, help='Nonlinear dimensions')
    parser.add_argument('--N', type=int, default=2, help='Output dimension (number of classes)')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate every N epochs')
    
    # Regularization hyperparameters
    parser.add_argument('--regularization', default=True, help='Use regularization')
    parser.add_argument('--tau', type=float, default=0.1, help='Regularization strength')
    parser.add_argument('--M_reg', type=int, default=5, help='Number of units to regularize')
    
    # Experiment hyperparameters
    parser.add_argument('--run_id', type=str, default='run_1', help='Run identifier')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    return parser.parse_args()

def save_results(model, metrics, args, save_dir):
    """Save model and metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f'model_{args.run_id}.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save metrics and hyperparameters
    results = {
        'metrics': metrics,
        'hyperparameters': vars(args)
    }
    metrics_path = os.path.join(save_dir, f'metrics_{args.run_id}.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    """Main function to run the experiment."""
    args = parse_args()
    set_thread_limits()
    
    # Run the experiment
    best_model, final_metrics = run_memory_task(
        # Task hyperparameters
        T=args.T,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        noise_std=args.noise_std,
        invert_prob=args.invert_prob,
        run_id=args.run_id,
        
        # Model hyperparameters
        M=args.M,
        L=args.L,
        N=args.N,
        input_dim=args.input_dim,
        
        # Training hyperparameters
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        
        # Regularization hyperparameters
        regularization=args.regularization,
        tau=args.tau,
        M_reg=args.M_reg
    )
    
    # Save results
    save_results(best_model, final_metrics, args, args.save_dir)
    
    print(f"Experiment completed. Results saved in {args.save_dir}")

if __name__ == '__main__':
    main() 