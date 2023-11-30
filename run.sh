#!/bin/bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 \
python lightning_run.py -w 4 -e 4 -g 4 -a 4 \
  --weight-round stochastic \
  --error-round nearest \
  --gradient-round nearest \
  --activation-round nearest \
  --learning-rate 0.1 \
  --momentum 0 \
  --batch-size 64 \
  # --checkpoint-path tb_logs/my_model/version_3/checkpoints/epoch=15-step=6256.ckpt\
