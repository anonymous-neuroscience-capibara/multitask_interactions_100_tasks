#!/usr/bin/env python
"""Rank foundation-sweep runs by mean best-epoch test accuracy.

Reads the per-run metadata.json files (written by train_foundation.py) under a
results directory and prints a ranked table: for each (M, N, P, sample, seed)
combo it shows the mean best-epoch test accuracy and the epoch at which the best
model was reached (`best_epoch`). Pure standard library -- runs with any Python
on the cluster (no torch / numpy / conda env needed).

Usage (from the project root, on the cluster):
    python "foundation model/rank_results.py"
    python "foundation model/rank_results.py" --results-dir "foundation model/results/foundation_model" --top 3
"""

import argparse
import glob
import json
import os
import re

# Folder fields are parsed individually so this works for BOTH naming schemes:
#   new:  foundation_joint_M256_N4_P8_s50_seed0      (no batch -> '_b' absent)
#   v2 :  foundation_joint_M256_b64_N4_P8_seed0      (has '_b64', no '_s')


def _from_name(folder, key):
    """Pull an int field like _M256 / _b64 / _N4 / _P8 / _s50 / seed0 from a name."""
    m = re.search(rf"{key}(\d+)", folder)
    return int(m.group(1)) if m else None


def _from_name_f(folder, key):
    """Pull a float field like _wd0.001 from a name."""
    m = re.search(rf"{key}([0-9.eE+-]+)", folder)
    return float(m.group(1)) if m else None


def load(meta_path):
    folder = os.path.basename(os.path.dirname(meta_path))
    try:
        with open(meta_path) as f:
            m = json.load(f)
    except Exception as e:
        return dict(folder=folder, error=str(e))

    return dict(
        folder=folder,
        # Prefer metadata (authoritative); fall back to the folder name.
        M=m.get("hidden_size", _from_name(folder, "_M")),
        N=m.get("nonlinear_units", _from_name(folder, "_N")),
        P=m.get(
            "num_individual_params", _from_name(folder, "_P")
        ),  # not in metadata yet
        sample=m.get("sample_size", _from_name(folder, "_s")),
        seed=m.get("seed", _from_name(folder, "seed")),
        # batch isn't saved in metadata; recover from '_b##' if present (v2 names).
        batch=m.get("batch_size", _from_name(folder, "_b")),
        wd=m.get("weight_decay", _from_name_f(folder, "_wd")),
        mean_acc=m.get("mean_best_test_accuracy"),
        best_epoch=m.get("best_epoch"),
        num_epochs=m.get("num_epochs"),
        error=None,
    )


def main():
    p = argparse.ArgumentParser(description="Rank foundation-sweep runs by accuracy.")
    p.add_argument("--results-dir", default="foundation model/results/foundation_model")
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args()

    metas = sorted(glob.glob(os.path.join(a.results_dir, "*", "metadata.json")))
    if not metas:
        print(f"No metadata.json found under {a.results_dir}")
        print("(Has the run finished? Each run folder should contain metadata.json.)")
        return

    rows = [load(m) for m in metas]
    ok = [r for r in rows if not r["error"] and r["mean_acc"] is not None]
    bad = [r for r in rows if r["error"] or r["mean_acc"] is None]
    ok.sort(key=lambda r: r["mean_acc"], reverse=True)

    def b(r):
        return str(r["batch"]) if r["batch"] is not None else "-"

    def wds(r):
        return ("%g" % r["wd"]) if r["wd"] is not None else "-"

    hdr = (
        f"{'rank':>4} {'M':>5} {'N':>3} {'P':>3} {'batch':>5} {'wd':>7} {'samp':>5} "
        f"{'seed':>4} {'mean_acc':>9} {'best_ep':>8} {'max_ep':>7}  folder"
    )
    print(hdr)
    print("-" * len(hdr))
    for rank, r in enumerate(ok, 1):
        print(
            f"{rank:>4} {str(r['M']):>5} {str(r['N']):>3} {str(r['P']):>3} "
            f"{b(r):>5} {wds(r):>7} {str(r['sample']):>5} {str(r['seed']):>4} "
            f"{r['mean_acc'] * 100:>8.2f}% {str(r['best_epoch']):>8} "
            f"{str(r['num_epochs']):>7}  {r['folder']}"
        )

    print(f"\n===== TOP {a.top} by mean best-epoch test accuracy =====")
    for rank, r in enumerate(ok[: a.top], 1):
        print(
            f"  #{rank}: {r['mean_acc'] * 100:.2f}%  reached at epoch "
            f"{r['best_epoch']} (of {r['num_epochs']})  |  "
            f"M={r['M']} N={r['N']} P={r['P']} batch={b(r)} wd={wds(r)} "
            f"sample={r['sample']} seed={r['seed']}"
        )
        print(f"        {r['folder']}")

    if bad:
        print(
            f"\n({len(bad)} run folder(s) had no usable metadata -- "
            f"unfinished or errored.)"
        )


if __name__ == "__main__":
    main()
