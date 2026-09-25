# launcher.py
import os
import multiprocessing


n_threads = 4
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["TORCH_NUM_THREADS"] = str(n_threads)

import subprocess
from multiprocessing import Pool

# User-configurable results folder
results_folder = "results_tanh"

Ls = [50]
Ms = [100]
taus = [0.1]
fixed_init_options = [False]
runs = range(1)
epochs = 75
nonlinearity_types = ["tanh"] #tanh, hardtanh, relu
rnn_types = ["plrnn"]

# Set your MNIST data root here if not default
mnist_data_root = "/export/home/mbrenner/Code/Linear_Nonlinear_Memory/classification_problems/mnist_data/MNIST"
#/export/home/mbrenner/Code/Linear_Nonlinear_Memory/classification_problems/mnist_data

def launch(args):
    L, tau, run, fixed_init, nonlinearity_type, rnn_type = args
    cmd = [
        "python", "run_one_config.py",
        "--dataset", "smnist",
        "--L", str(L),
        "--tau", str(tau),
        "--run", str(run),
        "--epochs", str(epochs),
        "--nonlinearity_type", nonlinearity_type,
        "--mnist_data_root", mnist_data_root,
        "--rnn_type", rnn_type,
        "--results_folder", results_folder,
    ]
    if fixed_init:
        cmd.append("--fixed_init")
    return subprocess.call(cmd)

if __name__ == "__main__":
    jobs = [
        (L, tau, run, fixed_init, nonlinearity_type, rnn_type)
        for L in Ls
        for tau in taus
        for run in runs
        for fixed_init in fixed_init_options
        for nonlinearity_type in nonlinearity_types
        for rnn_type in rnn_types
    ]
    with Pool(processes=35) as pool:  # Adjust number of workers (CPUs)
        pool.map(launch, jobs)
