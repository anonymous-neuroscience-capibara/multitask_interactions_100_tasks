import os
import subprocess
from multiprocessing import Pool

# Set thread limits
n_threads = 2
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
os.environ["TORCH_NUM_THREADS"] = str(n_threads)

# Parameters
Ls = [0, 1,3,5, 10, 20, 50, 75, 100]
#Ls = [0]
taus = [0.1]
fixed_init_options = [False]
runs = range(10)
epochs=15

# IMDb-specific config
glove_path = "add_your_path_here"  # Server

def launch(args):
    L, tau, run, fixed_init = args
    cmd = [
        "python", "run_one_config.py",
        "--dataset", "imdb",
        "--L", str(L),
        "--tau", str(tau),
        "--run", str(run),
        "--epochs", str(epochs),
        "--glove_path", glove_path
    ]
    if fixed_init:
        cmd.append("--fixed_init")

    return subprocess.call(cmd)

if __name__ == "__main__":
    jobs = [(L, tau, run, fixed_init) for L in Ls for tau in taus for run in runs for fixed_init in fixed_init_options]
    with Pool(processes=10) as pool:  # Adjust number of workers to your system
        pool.map(launch, jobs)

