#!/bin/bash
#SBATCH --job-name=compute_shaps
#SBATCH --output=/home/uh790919/AASXAI/output/compute_shaps_%A_%a.out
#SBATCH --array=1-10
#SBATCH --time=24:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --nodes=1

SCENARIO='BNSL-2016'

# Define the command to execute
SCRIPT_PATH=./computeShap_cluster_af.py
ARGUMENT="--scenario ${SCENARIO} --fold ${SLURM_ARRAY_TASK_ID}"

# Load any necessary modules

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate aasxai

# Execute the command
python ${SCRIPT_PATH} ${ARGUMENT}
