import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
import traceback

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from hierachical_model_task.bptt import BPTT
from torch.utils.data import DataLoader

from tasks import (
    CategoryDecision,
    ContextIntegration,
    DelayedMatchToSample,
    DelayedResponse,
    GoNogo,
)
from tasks.dataset import HierarchicalTasksDataset, collate_fn
from hierachical_model_task.utils import HiearchicalModelConfig

# Detect available CPU workers (cluster-aware)
def get_num_workers():
    """Detect number of workers from cluster scheduler or local CPU count."""
    # Check SLURM allocation first
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return max(1, int(os.environ['SLURM_CPUS_PER_TASK']) - 1)  # Reserve 1 for main
    # Check OMP_NUM_THREADS (common in HPC)
    if 'OMP_NUM_THREADS' in os.environ:
        return max(1, int(os.environ['OMP_NUM_THREADS']) - 1)
    # Fall back to local CPU count, capped at 8
    cpu_count = os.cpu_count() or 4
    return min(cpu_count - 1, 8)  # Reserve 1 CPU for main process, cap at 8

# Worker init function for deterministic DataLoader
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

duration_params = {
    'context': (5, 10),
    'stimulus': (10, 15),
    'delay': (10, 15),
    'response': (5, 10)
}

TASKS_MAP = {

    'DelayPro': DelayedResponse(duration_params, mode='pro'),
    'DelayAnti': DelayedResponse(duration_params, mode='anti'),
    'ReactPro': ReactionTime(duration_params, mode='pro'),
    'ReactAnti': ReactionTime(duration_params, mode='anti'),
    'CatPro': CategoryDecision(duration_params, mode='pro'),
    'CatAnti': CategoryDecision(duration_params, mode='anti'),
    'Match2Sample': DelayedMatchToSample(duration_params, mode='match'),
    'NonMatch2Sample': DelayedMatchToSample(duration_params, mode='nonmatch'),
    'CtxIntMod1': ContextIntegration(duration_params, relevant_modality=1),
    'CtxIntMod2': ContextIntegration(duration_params, relevant_modality=2),
    'GoNogo': GoNogo(duration_params),
    # "MotorPerturbation": MotorPerturbation(duration_params),
    # "PredictiveTracking": PredictiveTracking(duration_params)

}

task_names = list(TASKS_MAP.keys())


def _get_device(args: HiearchicalModelConfig):
    args.device = 'cpu'
    if args.use_gpu:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        try:
            args.device = args.device + ':' + str(args.device_id)
        except AttributeError:
            pass
    return args


def _handle_defaults(args: HiearchicalModelConfig):
    # set learning rates
    if args.individual_learning_rate is not None:
        args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    else:
        args.learning_rate = (args.learning_rate, args.learning_rate)
    # set teacher forcing alpha
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start
    return args


def run_single_experiment(num_tasks, num_individual_params, nonlinear_units, hidden_size, seed, output_dir):
    """
    Run a single experiment with specified number of tasks and individual parameters.
    
    Args:
        num_tasks: Number of tasks to use (1 to len(task_names))
        num_individual_params: Dimension of individual parameter vector
        nonlinear_units: Size of latent layer
        hidden_size: Size of hidden layer
        seed: Random seed for reproducibility
        output_dir: Directory to save results
    
    Returns:
        Dictionary with results
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Detect available workers and set thread count
    num_workers = get_num_workers()
    torch.set_num_threads(max(1, num_workers))

    # Select tasks
    selected_task_names = task_names[:num_tasks]
    tasks = [TASKS_MAP[name] for name in selected_task_names]

    print(f"Running experiment: tasks={num_tasks}, params={num_individual_params}, latent={nonlinear_units}, hidden={hidden_size}, seed={seed}")

    # Configure model
    args = HiearchicalModelConfig(tasks=tasks)
    args.num_individual_params = num_individual_params
    args.nonlinear_units = nonlinear_units
    args.hidden_size = hidden_size
    args.num_epochs = 300
    args = _get_device(args)
    args = _handle_defaults(args)

    # Create datasets
    train_dataset = HierarchicalTasksDataset(tasks, n_trials=len(tasks)*100)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    test_dataset = HierarchicalTasksDataset(
        tasks,
        n_trials=len(tasks)*50,
        task_indices=[i for i in range(len(tasks)) for _ in range(50)],
        fixed=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    # Train model
    training_alg = BPTT(None, args)
    loss_history, train_task_accuracies, test_loss_history, test_task_accuracies = training_alg.train(train_loader, test_loader)

    # Compute best metrics
    if test_loss_history:
        best_test_loss_idx = np.argmin(test_loss_history)
        best_test_loss = test_loss_history[best_test_loss_idx]
        # Get accuracy at the epoch with best test loss
        best_avg_test_accuracy = np.mean([
            test_task_accuracies[task_id][best_test_loss_idx] 
            for task_id in range(len(tasks))
        ])
    else:
        best_test_loss = float('inf')
        best_avg_test_accuracy = 0.0
    
    # Create output directory with new directory structure: hidden_size={}/seed={}/task{}_params_{}_latent_{}
    hidden_dir = output_dir / f"hidden_size_{hidden_size}"
    seed_dir = hidden_dir / f"seed_{seed}"
    experiment_name = f"tasks_{num_tasks}_params_{num_individual_params}_latent_{nonlinear_units}"
    experiment_dir = seed_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = experiment_dir / "model.pt"
    torch.save(training_alg.model.state_dict(), model_path)

    # Save training outputs
    results = {
        'num_tasks': num_tasks,
        'task_names': selected_task_names,
        'num_individual_params': num_individual_params,
        'nonlinear_units': nonlinear_units,
        'hidden_size': hidden_size,
        'seed': seed,
        'loss_history': loss_history,
        'test_loss_history': test_loss_history,
        'train_task_accuracies': train_task_accuracies,
        'test_task_accuracies': test_task_accuracies,
        'best_test_loss': best_test_loss,
        'best_avg_test_accuracy': best_avg_test_accuracy,
        'config': {
            'obs_size': args.obs_size,
            'hidden_size': args.hidden_size,
            'nonlinear_units': args.nonlinear_units,
            'M_reg': args.M_reg,
            'tau': args.tau,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate[0] if isinstance(args.learning_rate, tuple) else args.learning_rate,
            'individual_learning_rate': args.learning_rate[1] if isinstance(args.learning_rate, tuple) else args.learning_rate,
        }
    }

    # Save as numpy
    np.savez(
        experiment_dir / "results.npz",
        loss_history=np.array(loss_history),
        test_loss_history=np.array(test_loss_history),
        train_task_accuracies=np.array(train_task_accuracies),
        test_task_accuracies=np.array(test_task_accuracies)
    )

    # Save metadata as JSON
    metadata = {
        'num_tasks': num_tasks,
        'task_names': selected_task_names,
        'num_individual_params': num_individual_params,
        'nonlinear_units': nonlinear_units,
        'hidden_size': hidden_size,
        'seed': seed,
        'best_test_loss': best_test_loss,
        'best_avg_test_accuracy': best_avg_test_accuracy,
        'config': results['config'],
        'timestamp': datetime.now().isoformat()
    }
    with open(experiment_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return results


def main():
    # Create output directory
    output_dir = Path(__file__).parent / "results/reproducibility_experiments"
    output_dir.mkdir(exist_ok=True)

    total_tasks = len(task_names)
    all_results = []

    parser = argparse.ArgumentParser(description='Run a single hierarchical model experiment')
    parser.add_argument('--num_tasks', type=int, default=11, help='Number of tasks to use')
    parser.add_argument('--num_individual_params', type=int, required=True, help='Dimension of individual parameter vector')
    parser.add_argument('--nonlinear_units', type=int, required=True, help='Size of latent layer')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layer')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: results/reproducibility_experiments)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "results/reproducibility_experiments"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting experiment:")
    print(f"  num_tasks: {args.num_tasks}")
    print(f"  num_individual_params: {args.num_individual_params}")
    print(f"  nonlinear_units: {args.nonlinear_units}")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  seed: {args.seed}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*80}\n")
    
    try:
        # Run single experiment
        result = run_single_experiment(
            args.num_tasks,
            args.num_individual_params,
            args.nonlinear_units,
            args.hidden_size,
            args.seed,
            output_dir
        )
        
        if result is not None:
            print(f"\n{'='*80}")
            print(f"Experiment completed successfully!")
            print(f"  Best test loss: {result['best_test_loss']:.4f}")
            print(f"  Best avg test accuracy: {result['best_avg_test_accuracy']:.4f}")
            print(f"{'='*80}\n")
        else:
            print("\nExperiment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError running experiment: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()