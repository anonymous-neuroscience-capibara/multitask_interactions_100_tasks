import json
from pathlib import Path

import numpy as np
import torch
from helpers import save_model
from rnn_model import PLRNN, evaluate_model, train_multitask
from torch.utils.data import DataLoader

from tasks import (
    CategoryDecision,
    ContextIntegration,
    DelayedMatchToSample,
    DelayedResponse,
    GoNogo,
    ReactionTime,
)
from tasks.dataset import MultiTaskDataset, collate_fn

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DURATION_PARAMS_SIMPLE = {
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}

DURATION_PARAMS_HARD = {
    "context": (10, 20),
    "stimulus": (10, 40),
    "delay": (20, 50),
    "response": (10, 20),
}

TASK_NAMES = [
    "DelayPro",
    "DelayAnti",
    "ReactPro",
    "ReactAnti",
    "CatPro",
    "CatAnti",
    "Match2Sample",
    "NonMatch2Sample",
    "CtxIntMod1",
    "CtxIntMod2",
    "GoNogo",
]


def create_tasks(duration_params):
    return [
        DelayedResponse(duration_params, mode="pro"),
        DelayedResponse(duration_params, mode="anti"),
        ReactionTime(duration_params, mode="pro"),
        ReactionTime(duration_params, mode="anti"),
        CategoryDecision(duration_params, mode="pro"),
        CategoryDecision(duration_params, mode="anti"),
        DelayedMatchToSample(duration_params, mode="match"),
        DelayedMatchToSample(duration_params, mode="nonmatch"),
        ContextIntegration(duration_params, relevant_modality=1),
        ContextIntegration(duration_params, relevant_modality=2),
        GoNogo(duration_params),
    ]


def train_and_evaluate(
    M,
    L,
    N,
    duration_params,
    run_id=1,
    n_train_trials=5000,
    n_test_trials=1000,
    batch_size=64,
    num_epochs=150,
    lr=1e-3,
    tau=0.01,
    save_dir="checkpoints",
    results_dir="results",
):

    tasks = create_tasks(duration_params)

    train_dataset = MultiTaskDataset(tasks, n_trials=n_train_trials)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]

    model = PLRNN(M, L, N, input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_dataset = MultiTaskDataset(tasks, n_trials=n_test_trials)
    test_loader = DataLoader(
        test_dataset, batch_size=1000, shuffle=False, collate_fn=collate_fn
    )

    model.train()
    loss_history, train_task_accuracies, test_loss_history, test_task_accuracies = (
        train_multitask(
            model,
            train_loader,
            optimizer,
            device,
            tau=tau,
            M_reg=int(M / 2),
            num_epochs=num_epochs,
            test_loader=test_loader,
        )
    )

    test_accuracies = evaluate_model(model, test_loader, device, TASK_NAMES)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    model_name = f"11_tasks_model_M{M}_L{L}_N{N}_run{run_id}.pt"
    model_path = Path(save_dir) / model_name
    save_model(model, optimizer, str(model_path))

    results = {
        "M": M,
        "L": L,
        "N": N,
        "run_id": run_id,
        "duration_params": duration_params,
        "n_train_trials": n_train_trials,
        "n_test_trials": n_test_trials,
        "num_epochs": num_epochs,
        "lr": lr,
        "tau": tau,
        "model_path": str(model_path),
        "train_loss_history": [float(x) for x in loss_history],
        "test_loss_history": (
            [float(x) for x in test_loss_history] if test_loss_history else []
        ),
        "train_accuracies": {
            TASK_NAMES[i]: [float(x) for x in acc_list]
            for i, acc_list in train_task_accuracies.items()
        },
        "test_accuracies_history": (
            {
                TASK_NAMES[i]: [float(x) for x in acc_list]
                for i, acc_list in test_task_accuracies.items()
            }
            if test_task_accuracies and len(test_task_accuracies[0]) > 0
            else {}
        ),
        "test_accuracies": {
            TASK_NAMES[i]: float(acc) for i, acc in test_accuracies.items()
        },
        "mean_accuracy": float(np.mean(list(test_accuracies.values()))),
    }

    results_name = f"results_M{M}_L{L}_N{N}_run{run_id}.json"
    results_path = Path(results_dir) / results_name
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return model, results


if __name__ == "__main__":
    model, results = train_and_evaluate(
        M=64, L=32, N=3, duration_params=DURATION_PARAMS_SIMPLE, run_id=1
    )
    print(f"\nMean Test Accuracy: {results['mean_accuracy']:.4f}")
