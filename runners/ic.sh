#
# This script is used to run the training process for the HPA single-cell model.

export CUDA_VISIBLE_DEVICES=0

WORK_DIR=/home/unicamp/200208/models/hpa-single-cell-2nd

# Activate virtual environment if it exists
echo "Activating virtual environment... ($WORK_DIR/dev/bin/activate)"
source $WORK_DIR/dev/bin/activate

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# Train
$PY main_cp.py train -i r50_F0_cells -j jakiro/sin_exp5_r50d_rarex2.yaml
