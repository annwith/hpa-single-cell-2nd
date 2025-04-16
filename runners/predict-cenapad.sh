#PBS -N hpa-2-place
#PBS -q testegpu
#PBS -e logs/test.err
#PBS -o logs/test.log

ENV=cenapad
SCRATCH=$HOME
WORK_DIR=$HOME/hpa-single-cell-2nd

unset CUDA_VISIBLE_DEVICES
# export OMP_NUM_THREADS=8

module load python/3.8.11-gcc-9.4.0

# Activate virtual environment if it exists
echo "Activating virtual environment... ($HOME/hpa-single-cell-2nd/dev/bin/activate)"
source $HOME/hpa-single-cell-2nd/dev/bin/activate

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# Train
$PY predict.py train -i b3 -j jakiro/sin_exp5_b3_rare.yaml --predict_weights_path ../results/b3_F0/checkpoints/f0_epoch-18.pth
