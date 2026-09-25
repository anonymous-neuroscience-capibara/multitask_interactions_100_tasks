#!/bin/bash
#SBATCH --job-name=cmp_lstm
#SBATCH --output=/gs/home/garcias/Projects/multitask_learning/logs/cmp_lstm_%A_%a.out
#SBATCH --error=/gs/home/garcias/Projects/multitask_learning/logs/cmp_lstm_%A_%a.err
#SBATCH --partition=96GBXL
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=silvia.garcia@esi-frankfurt.de
#SBATCH --array=0-19

# 2026-08-11: hierarchical LSTM baseline, joint training on the 100-task battery,
# on x86 CPU (96GBXL, multitask env).
#   M{64,32} x P{8,16} x seed{0,1,2,3,4} = 2 x 2 x 5 = 20 -> --array=0-19
# M=64 is the state-dimension-matched arm; M=32 is the parameter-matched arm
# (4*32^2 = 64^2 recurrent params). No MAR (no analogue for gated cells; the
# script forces tau=0). Training config matches
# train_2026-08-06_M64_NPsweep_varnorm_extended_E880.sh (the final_run reference):
# LOSS_VARNORM=1, lr 5e-4, s200/b64, 3000 epochs, patience 300@300, warmup 100,
# w_init_gain 0.05, eval every epoch.
# DEVIATION: TEST=256 (reference used 50) -- larger test set for less noisy
# best-epoch selection; identical across all three comparison arms.

M_VALS=(64 32)
P_VALS=(8 16)
SEED_VALS=(0 1 2 3 4)

nM=${#M_VALS[@]}; nP=${#P_VALS[@]}; nSeed=${#SEED_VALS[@]}
i=$SLURM_ARRAY_TASK_ID
SEED=${SEED_VALS[$(( i % nSeed ))]}
PARAMS=${P_VALS[$(( (i / nSeed) % nP ))]}
HIDDEN=${M_VALS[$(( (i / (nSeed*nP)) % nM ))]}
# product = 2 M x 2 P x 5 seed = 20  ->  --array=0-19

# ---- fixed (identical to the extended varnorm NPsweep, so arms are comparable) ----
BATCH_LR=0.0005
SAMPLE=200
TEST=256
BATCH=64
EPOCHS=3000
ES_START=300
PATIENCE=300
WD=0.0001
GRADCLIP=10
LR_WARMUP=100
W_INIT_GAIN=0.05
INIT_MODE=default
EVAL_INTERVAL=1       # evaluate the test set every epoch (exact best_epoch)

OUTDIR="comparisons/results/joint/lstm_M${HIDDEN}_P${PARAMS}_s${SAMPLE}_b${BATCH}_lr${BATCH_LR}_seed${SEED}"

mkdir -p /gs/home/garcias/Projects/multitask_learning/logs
cd /gs/home/garcias/Projects/multitask_learning

echo "Job $SLURM_JOB_ID array task $i | Node: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"
echo "Config: LSTM M=$HIDDEN P=$PARAMS sample=$SAMPLE test=$TEST batch=$BATCH lr=$BATCH_LR epochs=$EPOCHS es=$PATIENCE@$ES_START eval=$EVAL_INTERVAL seed=$SEED  |  LOSS_VARNORM=1"
echo "Started at: $(date)"

# x86 node -> use the x86 multitask env, by ABSOLUTE PATH (do NOT conda activate)
PY=/gs/home/garcias/.conda/envs/multitask/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# match the final_run PLRNN reference: variance-normalized loss ON
export LOSS_VARNORM=1

"$PY" -u "comparisons/train_gated_foundation.py" \
    --cell lstm \
    --hidden-size "$HIDDEN" \
    --num-individual-params "$PARAMS" \
    --sample-size "$SAMPLE" \
    --test-size "$TEST" \
    --num-epochs "$EPOCHS" \
    --early-stopping-patience "$PATIENCE" \
    --early-stopping-start "$ES_START" \
    --batch-size "$BATCH" \
    --learning-rate "$BATCH_LR" \
    --weight-decay "$WD" \
    --grad-clip "$GRADCLIP" \
    --lr-warmup-epochs "$LR_WARMUP" \
    --w-init-gain "$W_INIT_GAIN" \
    --init-mode "$INIT_MODE" \
    --eval-interval "$EVAL_INTERVAL" \
    --seed "$SEED" \
    --output-dir "$OUTDIR"

echo "Finished at: $(date)"
