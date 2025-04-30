#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=48
#SBATCH -p sequana_gpu
#SBATCH -J b3-conf-f0
#SBATCH -o /scratch/lerdl/zanoni.dias/hpa-single-cell-2nd/logs/%j-hpa-project.out
#SBATCH -e /scratch/lerdl/zanoni.dias/hpa-single-cell-2nd/logs/%j-hpa-project.err
#SBATCH --time=96:00:00

#
# Train a model to perform multilabel classification.
#

echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

WORK_DIR=$SCRATCH/hpa-single-cell-2nd

module load gcc/9.3_sequana python/3.9.12_sequana cudnn/8.2_cuda-11.1_sequana

# Activate virtual environment if it exists
# echo "Activating virtual environment... ($SCRATCH/hpa-single-cell-2nd/dev/bin/activate)"
# source $SCRATCH/hpa-single-cell-2nd/dev/bin/activate

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

nvidia-smi  # For GPU memory
free -h      # For CPU memory

# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# Train
# $PY main.py train -i b3 -j jakiro/sin_exp5_b3_rare.yaml
$PY main_cp.py train -i b3_conf_aware_F0 -j jakiro/sin_exp5_b3_rare.yaml
