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

# Hyperparameter configurations for variable delay task
configs = {
    # Task hyperparameters
    'seq_lens': [5],
    'static_periods': [50],  # Fixed period before variable cue can occur
    'variable_periods': [50],  # Period during which cue can occur randomly
    'num_symbols': [5],
    
    # Model hyperparameters
    'Ls': [0,1,3,5,10,25,50],  # Nonlinear dimensions (only for PLRNN)
    'taus': [0.1],  # Regularization strengths (only for PLRNN)
    'Ms': [50],
    'model_types': ['plrnn'],  # Model types: 'plrnn', 'lstm', 'gru', 's4', 'hippo', 'mamba'
    'nonlinearity_types': ['relu'],  # Nonlinearity types: 'relu', 'tanh', 'gelu' (only for PLRNN)
    
    # Results organization
    'results_folder': 'results/results_variable_delay',  # Custom results folder name
    # Fixed hyperparameters
    'num_train': 1000,
    'num_test': 300,
    'time_dim': 0,
    'batch_size': 64,
    'num_epochs': 8000,
    'eval_every': 100,
    'learning_rate': 0.001,
    'include_time_encoding': False,
}

def launch(args):
    seq_len, static_period, variable_period, num_symbols, L, tau, M, model_type, nonlinearity_type, run = args
    
    # Create unique run identifier with clear model type and nonlinearity information
    if model_type == 'plrnn':
        run_id = f"run_{run}_plrnn_{nonlinearity_type}_L{L}_tau{tau:.3f}_seq{seq_len}_static{static_period}_var{variable_period}_sym{num_symbols}_M{M}"
    else:
        run_id = f"run_{run}_{model_type}_seq{seq_len}_static{static_period}_var{variable_period}_sym{num_symbols}_M{M}"
    
    cmd = [
        "python", "variable_delay_task.py",
        "--seq_len", str(seq_len),
        "--static_period", str(static_period),
        "--variable_period", str(variable_period),
        "--num_symbols", str(num_symbols),
        "--nonlin_size", str(L),  # This will be ignored for LSTM/GRU/SSMs but kept for compatibility
        "--tau", str(tau),  # This will be ignored for LSTM/GRU/SSMs but kept for compatibility
        "--run_id", run_id,
        "--hidden_size", str(M),  # Use M as the hidden_size parameter
        "--model_type", model_type,
        "--nonlinearity_type", nonlinearity_type,
        "--results_folder", configs['results_folder'],
        "--num_train", str(configs['num_train']),
        "--num_test", str(configs['num_test']),
        "--time_dim", str(configs['time_dim']),
        "--batch_size", str(configs['batch_size']),
        "--num_epochs", str(configs['num_epochs']),
        "--eval_every", str(configs['eval_every']),
        "--learning_rate", str(configs['learning_rate'])
    ]
    
    return subprocess.call(cmd)

if __name__ == "__main__":
    # Set up multiprocessing parameters
    n_threads = 1
    n_processes = 10
    n_runs = 5
    # Set thread limits
    set_thread_limits(n_threads)
    
    # Create all combinations of hyperparameters and runs
    jobs = list(itertools.product(
        configs['seq_lens'],
        configs['static_periods'],
        configs['variable_periods'],
        configs['num_symbols'],
        configs['Ls'],
        configs['taus'],
        configs['Ms'],
        configs['model_types'],
        configs['nonlinearity_types'],
        range(n_runs)
    ))
    
    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiment_configs", exist_ok=True)
    with open(f"experiment_configs/variable_delay_config_{timestamp}.json", 'w') as f:
        json.dump({
            'configs': configs,
            'n_threads': n_threads,
            'n_processes': n_processes,
            'n_runs': n_runs,
            'total_jobs': len(jobs)
        }, f, indent=4)
    
    # Launch jobs
    print(f"Launching {len(jobs)} total jobs with {n_processes} processes")
    print(f"Task: Variable Delay (static={configs['static_periods'][0]}, variable={configs['variable_periods'][0]})")
    print(f"Models: {configs['model_types']}")
    with Pool(processes=n_processes) as pool:
        pool.map(launch, jobs) 