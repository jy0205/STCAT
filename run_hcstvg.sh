#!/bin/bash

# Run for training
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/train_net.py \
 --config-file "experiments/HC-STVG/e2e_STCAT_R101_HCSTVG.yaml" \
 --use-seed \
 OUTPUT_DIR /home/tiger/data/hc-stvg/checkpoints/output \
 TENSORBOARD_DIR /home/tiger/data/hc-stvg/checkpoints/output/tensorboard \
 INPUT.RESOLUTION 448

# Run for Testing
# python3 -m torch.distributed.launch \
# --nproc_per_node=8 \
# scripts/test_net.py \
# --config-file "experiments/HC-STVG/e2e_STCAT_R101_HCSTVG.yaml" \
# --use-seed \
# MODEL.WEIGHT /home/tiger/data/hc-stvg/checkpoints/stcat_res448/model_012500.pth \
# OUTPUT_DIR /home/tiger/data/hc-stvg/checkpoints/output \
# INPUT.RESOLUTION 448