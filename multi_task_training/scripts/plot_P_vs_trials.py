import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""Plot heatmaps for P vs Trials using best_test_accuracies stored in NPZ files.

For every file in ../results/PvsTrials matching '*_training_history.npz' this script
loads the file, extracts keys like 'trials=10_P=1', reads the entry's
`best_test_accuracies` (expected to be a mapping from task index -> accuracy), and
builds a heatmap where x axis are trial values (sorted), y axis are the fixed task
names in the requested order, and values are per-task best test accuracies.

Saves one PNG per P value in the same PvsTrials folder: 'P=<P>_heatmap.png'.
"""

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


def load_file(filepath):
    try:
        data = dict(np.load(filepath, allow_pickle=True))
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")
    return data


def parse_key(key):
    # Expect format 'trials=<n>_P=<p>' (but be tolerant)
    parts = key.split("_")
    trials = None
    P = None
    for part in parts:
        if part.startswith("trials="):
            try:
                trials = int(part.split("=", 1)[1])
            except Exception:
                pass
        if part.startswith("P="):
            try:
                P = int(part.split("=", 1)[1])
            except Exception:
                pass
    return trials, P


def best_test_array_from_entry(entry, n_tasks=len(TASK_NAMES)):
    """Extract per-task best_test_accuracies from an entry's best_test_accuracies field.

    The saved value can be:
    - a dict mapping integers (0..n-1) -> floats,
    - an array-like sequence of length n_tasks,
    - or other pickled containers.

    Returns a numpy array length n_tasks with floats or np.nan for missing.
    """
    if entry is None:
        return np.full((n_tasks,), np.nan)

    # If it's an np.ndarray or list-like
    try:
        if isinstance(entry, (np.ndarray, list, tuple)):
            arr = np.asarray(entry)
            if arr.size == n_tasks:
                return arr.astype(float)
            # If it's 1D shorter/longer, try to map first n_tasks
            out = np.full((n_tasks,), np.nan)
            out[: min(n_tasks, arr.size)] = arr[: min(n_tasks, arr.size)]
            return out
    except Exception:
        pass

    # If it's a dict-like mapping
    try:
        d = dict(entry)
        out = np.full((n_tasks,), np.nan)
        for k, v in d.items():
            # ignore reserved keys if present
            if isinstance(k, str) and k.startswith("_"):
                continue
            try:
                idx = int(k)
            except Exception:
                # sometimes keys are stored as int in numpy object arrays
                continue
            if 0 <= idx < n_tasks:
                out[idx] = float(v)
        return out
    except Exception:
        pass

    # Fallback: try to coerce to float
    try:
        scalar = float(entry)
        return np.full((n_tasks,), scalar)
    except Exception:
        return np.full((n_tasks,), np.nan)


def plot_heatmap(matrix, x_labels, y_labels, title, outpath):
    plt.figure(figsize=(max(6, len(x_labels) * 0.6), max(6, len(y_labels) * 0.5)))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=x_labels,
        yticklabels=y_labels,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Trials")
    ax.set_ylabel("Task")
    ax.set_title(title)
    plt.tight_layout()
    # Show plot interactively instead of saving to disk
    plt.show()
    plt.close()


def main(pvs_dir=None):
    if pvs_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pvs_dir = os.path.join(base, "results", "PvsTrials")

    if not os.path.exists(pvs_dir):
        raise FileNotFoundError(pvs_dir)

    # Find training_history npz files
    files = [
        os.path.join(pvs_dir, f)
        for f in os.listdir(pvs_dir)
        if f.endswith("_training_history.npz")
    ]
    if not files:
        print(f"No '*_training_history.npz' files found in {pvs_dir}")
        return

    for filepath in files:
        data = load_file(filepath)

        # Collect available trial keys per file
        records = []  # list of tuples (trials, per_task_array)
        for key_bytes, value in data.items():
            # numpy may return bytes keys when saved in some cases
            key = (
                key_bytes.decode("utf-8")
                if isinstance(key_bytes, bytes)
                else str(key_bytes)
            )
            trials, P = parse_key(key)
            if trials is None:
                # skip keys we don't understand
                continue

            # value is the saved object for that trials,P combo
            try:
                entry = value.item() if hasattr(value, "item") else value
            except Exception:
                entry = value

            # some saved entries might already be dicts; extract best_test_accuracies
            best = None
            if isinstance(entry, dict):
                # prefer 'best_test_accuracies' key
                if "best_test_accuracies" in entry:
                    best = entry["best_test_accuracies"]
                elif "best_test_accuracy" in entry:
                    best = entry["best_test_accuracy"]
                else:
                    # maybe the entry is itself the accuracies mapping
                    best = entry
            else:
                # entry could be a numpy object that is e.g. a dict
                best = entry

            arr = best_test_array_from_entry(best, n_tasks=len(TASK_NAMES))
            records.append((trials, arr, P))

        if not records:
            print(f"No records parsed from {filepath}")
            continue

        # Determine P for this file (prefer the P parsed from filename or entries)
        # Attempt to get P from filename like 'P=1_training_history.npz'
        fname = os.path.basename(filepath)
        P_from_name = None
        if fname.startswith("P="):
            try:
                P_from_name = int(fname.split("_", 1)[0].split("=")[1])
            except Exception:
                P_from_name = None

        P_values = set([r[2] for r in records if r[2] is not None])
        P_val = (
            P_from_name
            if P_from_name is not None
            else (list(P_values)[0] if P_values else "unknown")
        )

        # Build matrix: columns sorted by trials
        records.sort(key=lambda x: x[0])
        trials_list = [r[0] for r in records]
        matrix = np.stack([r[1] for r in records], axis=1)  # shape (n_tasks, n_trials)

        # Plot with trials on x axis and tasks on y axis
        outname = os.path.join(pvs_dir, f"P={P_val}_heatmap.png")
        title = f"P={P_val} best test accuracies (per task)"
        plot_heatmap(
            matrix,
            x_labels=trials_list,
            y_labels=TASK_NAMES,
            title=title,
            outpath=outname,
        )
        print(f"Saved heatmap for P={P_val} -> {outname}")


if __name__ == "__main__":
    main()
