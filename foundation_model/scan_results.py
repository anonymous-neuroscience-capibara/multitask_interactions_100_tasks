#!/usr/bin/env python
"""Scan SLURM .out logs from a foundation sweep and rank the combos by test
accuracy -- using ONLY the .out files (no model.pt / results.npz needed).

Each .out has a `Config:` line and either a final SUMMARY block
(`best-epoch test accuracy (mean XX.X%)`) if the run finished, or per-epoch
`Test Acc: ..%` lines if it's still running / was killed. This reads whichever
is available and prints a ranked table.

Usage:
    python "foundation model/scan_results.py" [--logs-dir logs]
Run it on the cluster (where the logs live), or after rsync-ing logs locally.
"""

import argparse
import glob
import os
import re

CONFIG_RE = re.compile(
    r"Config:\s*M=(\d+)\s+n=(\d+)\s+P=(\d+)\s+batch=(\d+).*?seed=(\d+)"
)
SUMMARY_RE = re.compile(r"best-epoch test accuracy \(mean\s+([\d.]+)%\)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+.*?Test Acc:\s*([\d.]+)%")


def parse(path):
    with open(path, errors="ignore") as f:
        text = f.read()
    cfg = CONFIG_RE.search(text)
    config = None
    if cfg:
        M, n, P, b, seed = cfg.groups()
        config = dict(M=int(M), n=int(n), P=int(P), batch=int(b), seed=int(seed))

    summary = SUMMARY_RE.search(text)
    epochs = EPOCH_RE.findall(text)  # list of (epoch, test_acc%)
    if summary:
        status, mean_acc = "done", float(summary.group(1))
        last_epoch = int(epochs[-1][0]) if epochs else None
    elif epochs:
        # Unfinished: best mean test acc seen so far across logged epochs.
        mean_acc = max(float(a) for _, a in epochs)
        last_epoch = int(epochs[-1][0])
        status = f"running@{last_epoch}"
    else:
        mean_acc, last_epoch, status = None, None, "no-eval-yet"
    return dict(
        file=os.path.basename(path),
        config=config,
        mean_acc=mean_acc,
        last_epoch=last_epoch,
        status=status,
    )


def main():
    p = argparse.ArgumentParser(description="Rank a foundation sweep from .out logs.")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--glob", default="slurm_*.out")
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.logs_dir, a.glob)))
    if not paths:
        print(f"No logs matching {a.glob} in {a.logs_dir}")
        return

    rows = [parse(p) for p in paths]
    rows = [r for r in rows if r["config"] is not None]
    # Rank by mean accuracy (None last).
    rows.sort(
        key=lambda r: (r["mean_acc"] is not None, r["mean_acc"] or -1), reverse=True
    )

    hdr = f"{'M':>5} {'n':>3} {'P':>3} {'batch':>5} {'seed':>4} {'status':>12} {'mean_test_acc':>14}  file"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        c = r["config"]
        acc = f"{r['mean_acc']:.1f}%" if r["mean_acc"] is not None else "--"
        print(
            f"{c['M']:>5} {c['n']:>3} {c['P']:>3} {c['batch']:>5} {c['seed']:>4} "
            f"{r['status']:>12} {acc:>14}  {r['file']}"
        )

    done = [r for r in rows if r["mean_acc"] is not None]
    if done:
        best = done[0]
        c = best["config"]
        print(
            f"\nBEST so far: M={c['M']} n={c['n']} P={c['P']} batch={c['batch']} "
            f"seed={c['seed']}  ->  {best['mean_acc']:.1f}% ({best['status']})"
        )
        print("  (metrics from .out; the trained weights are that run's model.pt)")


if __name__ == "__main__":
    main()
