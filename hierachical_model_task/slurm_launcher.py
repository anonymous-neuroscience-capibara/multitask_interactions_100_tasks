#!/usr/bin/env python3
"""
SLURM Launcher for Hierarchical Model Experiments

This script generates and optionally submits SLURM job scripts for running
hierarchical model experiments. Each job runs a single experiment configuration.

Usage:
    python slurm_launcher.py --submit  # Generate and submit jobs
    python slurm_launcher.py          # Only generate job scripts (dry run)
"""

import itertools
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import argparse


def create_slurm_script(
    job_name,
    num_tasks,
    num_individual_params,
    latent_size,
    hidden_size,
    seed,
    output_dir,
    slurm_config,
    script_dir,
    project_dir,
):
    """
    Create a SLURM job script for a single experiment.
    
    Args:
        job_name: Name for the SLURM job
        num_tasks: Number of tasks
        num_individual_params: Dimension of individual parameter vector
        latent_size: Size of latent layer
        hidden_size: Size of hidden layer
        seed: Random seed
        output_dir: Directory to save results
        slurm_config: Dictionary with SLURM configuration (time, mem, cpus, etc.)
        script_dir: Directory to save job scripts
        project_dir: Root directory of the project
    
    Returns:
        Path to the created job script
    """
    
    script_path = script_dir / f"{job_name}.sh"
    
    # Build the python command with absolute path
    python_script = project_dir / "run_experiments_v2.py"
    python_cmd = (
        f"python {python_script} "
        f"--num_tasks {num_tasks} "
        f"--num_individual_params {num_individual_params} "
        f"--latent_size {latent_size} "
        f"--hidden_size {hidden_size} "
        f"--seed {seed} "
        f"--output_dir {output_dir}"
    )
    
    # Create SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --cpus-per-task={slurm_config['cpus']}
#SBATCH --mem={slurm_config['mem']}"""
    
    # Add optional SLURM parameters
    if slurm_config.get('time'):
        script_content += f"\n#SBATCH --time={slurm_config['time']}"
    if slurm_config.get('partition'):
        script_content += f"\n#SBATCH --partition={slurm_config['partition']}"
    if slurm_config.get('account'):
        script_content += f"\n#SBATCH --account={slurm_config['account']}"
    if slurm_config.get('qos'):
        script_content += f"\n#SBATCH --qos={slurm_config['qos']}"
    if slurm_config.get('gres'):
        script_content += f"\n#SBATCH --gres={slurm_config['gres']}"
    
    script_content += f"""


# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Started at: $(date)"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {slurm_config['conda_env']}

# Set environment variables for reproducibility and optimal CPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to script directory
cd $SLURM_SUBMIT_DIR || {{ echo "Failed to cd to submit directory"; exit 1; }}

# Run the experiment
echo "Running experiment: tasks={num_tasks}, params={num_individual_params}, latent={latent_size}, hidden={hidden_size}, seed={seed}"
{python_cmd} || {{ echo "Python crashed with exit code $?"; exit 1; }}

# Print completion info
echo "Finished at: $(date)"
"""
    
    # Write script to file
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts for hierarchical model experiments')
    parser.add_argument('--submit', action='store_true', help='Submit jobs after generating scripts')
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON file')
    args = parser.parse_args()
    
    # Project directory
    project_dir = Path(__file__).parent.resolve()
    
    # Default configuration
    config = {
        "num_tasks": 11,
        "seeds": [0, 1, 2],
        "latent_sizes": [1, 2, 3, 4, 5, 6, 8],
        "hidden_sizes": [32, 64],
        "num_individual_params_list": [1, 2, 3, 4, 8, 11, 12],
        "output_dir": "results/reproducibility_experiments",
        "slurm": {
            "time": None,     # Set if time limit needed, e.g., "4:00:00"
            "mem": "48GB",
            "cpus": 1,
            "partition": None,  # Set if required by your cluster
            "conda_env": "psynamic",
            "account": None,  # Set if required by your cluster
            "qos": None,      # Set if required by your cluster
            "gres": None      # e.g., "gpu:1" if using GPUs
        }
    }
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            # Update config with user values
            config.update(user_config)
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = project_dir / f"slurm_jobs_{timestamp}"
    script_dir.mkdir(exist_ok=True)
    (script_dir / "logs").mkdir(exist_ok=True)
    
    # Save configuration
    with open(script_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate all experiment configurations
    jobs = list(itertools.product(
        [config["num_tasks"]],
        config["num_individual_params_list"],
        config["latent_sizes"],
        config["hidden_sizes"],
        config["seeds"]
    ))
    
    print(f"{'='*80}")
    print(f"SLURM Job Generator for Hierarchical Model Experiments")
    print(f"{'='*80}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Script directory: {script_dir}")
    print(f"Submit jobs: {args.submit}")
    print(f"{'='*80}\n")
    
    # Generate job scripts
    job_scripts = []
    for num_tasks, num_params, latent_size, hidden_size, seed in jobs:
        job_name = f"hier_t{num_tasks}_p{num_params}_l{latent_size}_h{hidden_size}_s{seed}"
        
        script_path = create_slurm_script(
            job_name=job_name,
            num_tasks=num_tasks,
            num_individual_params=num_params,
            latent_size=latent_size,
            hidden_size=hidden_size,
            seed=seed,
            output_dir=config["output_dir"],
            slurm_config=config["slurm"],
            script_dir=script_dir,
            project_dir=project_dir
        )
        
        job_scripts.append((job_name, script_path))
    
    print(f"Generated {len(job_scripts)} job scripts in {script_dir}\n")
    
    # Submit jobs if requested
    if args.submit:
        print("Submitting jobs...")
        submission_log = []
        
        for job_name, script_path in job_scripts:
            try:
                result = subprocess.run(
                    ['sbatch', str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted {job_name}: Job ID {job_id}")
                submission_log.append({
                    'job_name': job_name,
                    'job_id': job_id,
                    'script_path': str(script_path),
                    'status': 'submitted'
                })
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {job_name}: {e}")
                submission_log.append({
                    'job_name': job_name,
                    'script_path': str(script_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save submission log
        with open(script_dir / "submission_log.json", 'w') as f:
            json.dump(submission_log, f, indent=2)
        
        print(f"\nSubmitted {sum(1 for s in submission_log if s['status'] == 'submitted')}/{len(job_scripts)} jobs successfully")
    else:
        print("Dry run completed. Use --submit to submit jobs to SLURM.")
        print(f"\nTo submit all jobs manually, run:")
        print(f"  cd {script_dir}")
        print(f"  for script in *.sh; do sbatch $script; done")
    
    print(f"\nJob scripts saved in: {script_dir}")
    print(f"Logs will be saved in: {script_dir}/logs/")


if __name__ == '__main__':
    main()
