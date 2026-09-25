#!/bin/bash
#SBATCH --job-name=multitask_exp
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=silvia.garcia@esi-frankfurt.de

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Started at: $(date)"

# Load conda/module if needed (adjust for your cluster)
# module load anaconda3  # Uncomment if your cluster uses modules

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate multitask

# Set environment variables for reproducibility
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to script directory
cd $SLURM_SUBMIT_DIR

# Run the experiment (each array task uses its index as the seed)
SEED=$SLURM_ARRAY_TASK_ID
python run_experiments.py --seeds $SEED

# Print completion info
echo "Finished at: $(date)"
