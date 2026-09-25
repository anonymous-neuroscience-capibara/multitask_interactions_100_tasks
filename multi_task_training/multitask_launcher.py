import itertools
import json
import os
from datetime import datetime
from multiprocessing import Pool

from train_models import (
    DURATION_PARAMS_HARD,
    DURATION_PARAMS_SIMPLE,
    train_and_evaluate,
)


def set_thread_limits(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["TORCH_NUM_THREADS"] = str(n_threads)


DURATION_MAP = {"simple": DURATION_PARAMS_SIMPLE, "hard": DURATION_PARAMS_HARD}

configs = {
    "Ms": [64],
    "Ls": [0, 1, 2, 4, 8, 16, 32, 64],
    "N": 3,
    "duration_modes": ["hard"],
    "taus": [0.01],
    "n_train_trials": 5000,
    "n_test_trials": 1000,
    "batch_size": 64,
    "num_epochs": 300,
    "lr": 1e-3,
    "save_dir": "checkpoints",
    "results_dir": "results",
}


def launch(args):
    M, L, duration_mode, tau, run = args
    duration_params = DURATION_MAP[duration_mode]

    try:
        train_and_evaluate(
            M=M,
            L=L,
            N=configs["N"],
            duration_params=duration_params,
            run_id=run,
            n_train_trials=configs["n_train_trials"],
            n_test_trials=configs["n_test_trials"],
            batch_size=configs["batch_size"],
            num_epochs=configs["num_epochs"],
            lr=configs["lr"],
            tau=tau,
            save_dir=configs["save_dir"],
            results_dir=configs["results_dir"],
        )
        return 0
    except Exception as e:
        print(
            f"Error in run {run} (M={M}, L={L}, mode={duration_mode}, tau={tau}): {e}"
        )
        return 1


if __name__ == "__main__":
    n_threads = 1
    n_processes = 20
    n_runs = 5

    set_thread_limits(n_threads)

    jobs = list(
        itertools.product(
            configs["Ms"],
            configs["Ls"],
            configs["duration_modes"],
            configs["taus"],
            range(1, n_runs + 1),
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiment_configs", exist_ok=True)
    with open(f"experiment_configs/config_{timestamp}.json", "w") as f:
        json.dump(
            {
                "configs": configs,
                "n_threads": n_threads,
                "n_processes": n_processes,
                "n_runs": n_runs,
                "total_jobs": len(jobs),
            },
            f,
            indent=2,
        )

    print(f"Launching {len(jobs)} jobs with {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        results = pool.map(launch, jobs)

    print(f"Completed: {sum(1 for r in results if r == 0)}/{len(results)} successful")
