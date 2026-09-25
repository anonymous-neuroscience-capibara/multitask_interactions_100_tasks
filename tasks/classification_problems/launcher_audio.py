import os
import subprocess
from multiprocessing import Pool

# Set thread limits
n_threads = 8
thread_limit = False
if thread_limit:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["TORCH_NUM_THREADS"] = str(n_threads)

# Parameters
Ls = [50]
taus = [0.1]
fixed_init_options = [False]
runs = range(1)

# Dataset-specific paths
train_path = "add_your_path_here"
test_path  = "add_your_path_here"

def launch(args):
    L, tau, run, fixed_init = args
    cmd = [
        "python", "run_one_config.py",
        "--dataset", "speech",
        "--L", str(L),
        "--tau", str(tau),
        "--run", str(run),
        "--audio_train_path", train_path,
        "--audio_test_path", test_path
    ]
    if fixed_init:
        cmd.append("--fixed_init")

    return subprocess.call(cmd)

if __name__ == "__main__":
    jobs = [(L, tau, run, fixed_init) for L in Ls for tau in taus for run in runs for fixed_init in fixed_init_options]
    with Pool(processes=1) as pool:
        pool.map(launch, jobs)
