from pathlib import Path

import torch
from rnn_model import PLRNN


def save_model(model, optimizer, filepath):  # loss_history, task_accuracies,
    """Save model checkpoint"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            #  'loss_history': loss_history,
            # 'task_accuracies': task_accuracies,
            "model_config": {
                "M": model.M,
                "L": model.L,
                "N": model.N,
                "input_dim": model.input_dim,
            },
        },
        filepath,
    )


def load_model(filepath, model=None, optimizer=None, device="cpu"):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint["model_config"]
    model = PLRNN(cfg["M"], cfg["L"], cfg["N"], cfg["input_dim"]).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        model,
        optimizer,
    )  # , checkpoint['loss_history'], checkpoint['task_accuracies']
