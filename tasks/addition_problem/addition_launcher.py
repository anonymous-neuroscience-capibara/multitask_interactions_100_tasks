import os
import subprocess
from multiprocessing import Pool
import itertools
import json
from datetime import datetime

# Set thread limits
def set_thread_limits(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["TORCH_NUM_THREADS"] = str(n_threads)

# Hyperparameter configurations
configs = {
    # Task hyperparameters
    'Ts': [100],  # Sequence lengths
    # Model hyperparameters
    'Ls': [50],  # Nonlinear dimensions
    'taus': [0.1],  # Regularization strengths for A
    
    # Fixed hyperparameters
    'M': 50,  # Hidden size
    'N': 1,   # Output dimension (sum)
    'input_dim': 2,  # Input dimension (value + mask)
    'M_reg': 25,  # Number of units for regularization
    'num_train': 1000,
    'num_test': 1000,
    'batch_size': 128,
    'num_epochs': 2000,
    'eval_every': 100,
    'learning_rate': 0.001,
    'regularization': True
}

def launch(args):
    T, L, tau, run = args
    
    # Create unique run identifier
    run_id = f"run_{run}_T{T}_L{L}_tau{tau:.3f}"
    
    # Create save directory
    save_dir = os.path.join("results", "addition_task", run_id)
    
    # Create training hyperparameters dictionary (without save_dir)
    train_params = {
        'T': T,
        'L': L,
        'tau': tau,
        'M': configs['M'],
        'N': configs['N'],
        'input_dim': configs['input_dim'],
        'M_reg': configs['M_reg'],
        'num_train': configs['num_train'],
        'num_test': configs['num_test'],
        'batch_size': configs['batch_size'],
        'num_epochs': configs['num_epochs'],
        'eval_every': configs['eval_every'],
        'learning_rate': configs['learning_rate'],
        'regularization': configs['regularization'],
        'run_id': run_id
    }
    
    # Generate command line arguments
    cmd = ["python", "addition_task.py"]
    # Add training parameters
    for key, value in train_params.items():
        cmd.extend([f"--{key}", str(value)])
    # Add save directory separately
    cmd.extend(["--save_dir", save_dir])
    
    return subprocess.call(cmd)

if __name__ == "__main__":
    # Set up multiprocessing parameters
    n_threads = 1
    n_processes = 5
    n_runs = 5
    
    # Set thread limits
    set_thread_limits(n_threads)
    
    # Create all combinations of hyperparameters and runs
    jobs = list(itertools.product(
        configs['Ts'],
        configs['Ls'],
        configs['taus'],
        range(n_runs)
    ))
    
    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiment_configs", exist_ok=True)
    with open(f"experiment_configs/config_{timestamp}.json", 'w') as f:
        json.dump({
            'configs': configs,
            'n_threads': n_threads,
            'n_processes': n_processes,
            'n_runs': n_runs,
            'total_jobs': len(jobs)
        }, f, indent=4)
    
    # Launch jobs
    print(f"Launching {len(jobs)} total jobs with {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        pool.map(launch, jobs)
