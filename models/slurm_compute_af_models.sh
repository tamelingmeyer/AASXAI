#!/bin/bash
#SBATCH --job-name=compute_shaps
#SBATCH --output=./compute_shaps_%A_%a.out
#SBATCH --array=1-10
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --nodes=1

SCENARIO="BNSL-2016"
LIMIT="86400"
#LIMIT=10
#SLURM_ARRAY_TASK_ID=1


# Load any necessary modules

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate aasxai

# Execute the command
mkdir ./models/${SCENARIO}/

python ./AutoFolio/scripts/autofolio -s ./aslib_data/${SCENARIO} --outer-cv --outer-cv-fold ${SLURM_ARRAY_TASK_ID} -t --wallclock_limit ${LIMIT} --runcount_limit 1000000 --output_dir ./models/${SCENARIO}/${SLURM_ARRAY_TASK_ID}_smac --out-template ./models/${SCENARIO}/'${fold}.${type}'
