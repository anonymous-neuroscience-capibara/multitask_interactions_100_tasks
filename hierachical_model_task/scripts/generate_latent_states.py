"""Generate latent states and bitcodes for all experiments under data/20260331_CD."""

import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\garcias\Documents\Projects\Linear_Nonlinear_Memory\data\20260331_AW\20260331_AW"
)
SCRIPT = Path(
    r"C:\Users\garcias\Documents\Projects\Linear_Nonlinear_Memory\hierachical_model_task\run_experiments.py"
)
SEEDS = [0, 1, 2, 3]
HIERARCHISATION = "CD"  # this dataset was trained with CD-only hierarchisation

EXP_PATTERN = re.compile(r"^p(\d+)_n(\d+)_h(\d+)_s(\d+)$")

for seed in SEEDS:
    seed_dir = DATA_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        print(f"Skipping seed_{seed}: directory not found")
        continue

    for exp_dir in sorted(seed_dir.glob("p*_n*_h*_s*")):
        if not exp_dir.is_dir():
            continue

        m = EXP_PATTERN.match(exp_dir.name)
        if not m:
            continue

        sample_size = int(m.group(4))

        if (exp_dir / "latent_and_bitcodes").exists():
            print(f"Skipping {exp_dir.name} (seed_{seed}) - already done")
            continue

        print(
            f"Processing: seed_{seed}/{exp_dir.name} (sample_size={sample_size}, hier={HIERARCHISATION})"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--load-model",
                str(exp_dir),
                "--sample-size",
                str(sample_size),
                "--hierarchisation",
                HIERARCHISATION,
            ],
        )
        if result.returncode != 0:
            print(f"  ERROR (exit code {result.returncode})")

print("Done generating latent states.")
