#!/usr/bin/env python
"""Find missing experiment combinations by reading parameters from run_slurm.sh."""

import argparse
import re
from pathlib import Path


def parse_bash_array(slurm_text, var_name):
    """Extract a bash array like VAR=(1 2 3) from shell script text."""
    pattern = rf"{var_name}=\(([^)]+)\)"
    match = re.search(pattern, slurm_text)
    if not match:
        raise ValueError(f"Could not find {var_name}=(...) in slurm script")
    return [int(x) for x in match.group(1).split()]


def main():
    parser = argparse.ArgumentParser(description="Find missing experiment folders")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to results directory containing seed_* folders",
    )
    parser.add_argument(
        "--slurm-script",
        type=str,
        default=None,
        help="Path to run_slurm.sh (default: run_slurm.sh in project root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: missing_combinations.txt next to this script)",
    )
    args = parser.parse_args()

    # Find and parse slurm script
    if args.slurm_script is not None:
        slurm_path = Path(args.slurm_script)
    else:
        slurm_path = Path(__file__).parent.parent / "run_slurm.sh"

    slurm_text = slurm_path.read_text()

    seeds = parse_bash_array(slurm_text, "SEEDS")
    hidden_sizes = parse_bash_array(slurm_text, "HIDDEN_SIZES")
    num_individual_params_list = parse_bash_array(slurm_text, "INDIVIDUAL_PARAMS")
    nonlinear_units_list = parse_bash_array(slurm_text, "NONLINEAR_UNITS")
    sample_sizes = parse_bash_array(slurm_text, "SAMPLE_SIZES")

    print(f"Parsed from {slurm_path}:")
    print(f"  Seeds: {seeds}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Individual params: {num_individual_params_list}")
    print(f"  Nonlinear units: {nonlinear_units_list}")
    print(f"  Sample sizes: {sample_sizes}")
    print()

    base_path = Path(args.results_dir)

    # Generate all expected folder names
    expected_folders = set()
    for hidden_size in hidden_sizes:
        for num_params in num_individual_params_list:
            for nonlinear in nonlinear_units_list:
                for s in sample_sizes:
                    expected_folders.add(
                        f"p{num_params}_n{nonlinear}_h{hidden_size}_s{s}"
                    )

    print(f"Total expected folders per seed: {len(expected_folders)}\n")

    # Check missing folders per seed
    total_missing = {}
    for seed in seeds:
        seed_path = base_path / f"seed_{seed}"
        actual_folders = set()

        if seed_path.exists():
            for item in seed_path.iterdir():
                if item.is_dir():
                    actual_folders.add(item.name)

        missing_folders = expected_folders - actual_folders
        missing_folders = sorted(
            missing_folders,
            key=lambda x: (
                int(x.split("_")[0][1:]),
                int(x.split("_")[1][1:]),
                int(x.split("_")[3][1:]),
            ),
        )
        total_missing[seed] = missing_folders
        print(
            f"Seed {seed}: {len(actual_folders)} actual, {len(missing_folders)} missing"
        )

    # Write flat file: one line per combo (seed params nonlinear hidden sample_size)
    if args.output is not None:
        output_file = Path(args.output)
    else:
        output_file = Path(__file__).parent / "missing_combinations.txt"

    total_combos = 0
    with open(output_file, "w") as f:
        for seed, folders in sorted(total_missing.items()):
            for folder in folders:
                parts = folder.split("_")
                p = parts[0][1:]
                n = parts[1][1:]
                h = parts[2][1:]
                s = parts[3][1:]
                f.write(f"{seed} {p} {n} {h} {s}\n")
                total_combos += 1

    print(f"\nWritten {total_combos} combos to {output_file}")
    print(f"Use: sbatch --array=0-{total_combos - 1} run_slurm.sh")


if __name__ == "__main__":
    main()
