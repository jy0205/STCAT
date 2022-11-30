#!/bin/bash

# # Run for training
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/train_net.py \
 --config-file "experiments/VidSTG/e2e_STCAT_R101_VidSTG.yaml" \
 --use-seed \
 OUTPUT_DIR data/vidstg/checkpoints/output \
 TENSORBOARD_DIR data/vidstg/checkpoints/output/tensorboard \
 INPUT.RESOLUTION 448

# Run for Testing
# python3 -m torch.distributed.launch \
# --nproc_per_node=8 \
# scripts/test_net.py \
# --config-file "experiments/VidSTG/e2e_STCAT_R101_VidSTG.yaml" \
# --use-seed \
# MODEL.WEIGHT data/vidstg/checkpoints/stcat_res448/vidstg_res448.pth \
# OUTPUT_DIR data/vidstg/checkpoints/output \
# INPUT.RESOLUTION 448