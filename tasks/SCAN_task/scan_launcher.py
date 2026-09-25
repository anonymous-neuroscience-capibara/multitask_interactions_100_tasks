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
    # Model hyperparameters
    'L_encoders': [0, 4,8,16, 64, 128],  # Nonlinear dimensions for encoder
    'L_decoders': [32],  # Nonlinear dimensions for decoder
    
    # Fixed hyperparameters
    'M': 128,  # Hidden size
    'embedding_dim': 64,  # Input embedding dimension
    'output_embedding_dim': 64,  # Output embedding dimension
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'regularization': True,
    'tau': 0.1,
    'M_reg': 64,
    'initial_teacher_forcing': 0.0, #results were better without teacher forcing
    'end_token_weight': 5.0
}

def launch(args):
    L_encoder, L_decoder, run = args
    
    # Create unique run identifier
    run_id = f"run_{run}_Lenc{L_encoder}_Ldec{L_decoder}"
    
    # Create save directory
    save_dir = os.path.join("results", "scan_task", run_id)
    
    # Create training hyperparameters dictionary
    train_params = {
        'L_encoder': L_encoder,
        'L_decoder': L_decoder,
        'M': configs['M'],
        'embedding_dim': configs['embedding_dim'],
        'output_embedding_dim': configs['output_embedding_dim'],
        'batch_size': configs['batch_size'],
        'num_epochs': configs['num_epochs'],
        'learning_rate': configs['learning_rate'],
        'regularization': configs['regularization'],
        'tau': configs['tau'],
        'M_reg': configs['M_reg'],
        'initial_teacher_forcing': configs['initial_teacher_forcing'],
        'end_token_weight': configs['end_token_weight'],
        'save_dir': save_dir
    }
    
    # Generate command line arguments
    cmd = ["python", "main.py"]
    # Add training parameters
    for key, value in train_params.items():
        cmd.extend([f"--{key}", str(value)])
    
    return subprocess.call(cmd)

if __name__ == "__main__":
    # Set up multiprocessing parameters
    n_threads = 1
    n_processes = 35
    n_runs = 10
    
    # Set thread limits
    set_thread_limits(n_threads)
    
    # Create all combinations of hyperparameters and runs
    jobs = list(itertools.product(
        configs['L_encoders'],
        configs['L_decoders'],
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
