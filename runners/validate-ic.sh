#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

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

# Set variables
MODEL=b3
CFG=jakiro/sin_exp5_b3_rare.yaml
PREDICT_WEIGHTS_PATH=/home/unicamp/200208/results/b3_F2/checkpoints
VAL_REPORT_TXT=/home/unicamp/200208/results/b3_F2/val_report.txt

# Run validation
$PY basic_validate.py \
    valid \
    -i $MODEL \
    -j $CFG \
    --predict_weights_path $PREDICT_WEIGHTS_PATH \
    --val_report_txt $VAL_REPORT_TXT