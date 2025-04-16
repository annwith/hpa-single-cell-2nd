# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# Train
$PY predict.py train -i b3 -j jakiro/sin_exp5_b3_rare.yaml --predict_weights_path ../results/b3_F0/checkpoints/f0_epoch-18.pth
