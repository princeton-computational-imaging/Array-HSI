#!/bin/bash

# Saving and logging
RESULT_DIR=[path to save results and ckpts]
SAVE_FREQ=20
EPOCH=400

# Training
FORWARD_MODEL=optical_forward
GENERATOR=HR_UNet
PATCH_SIZE=512
H_LR=1e-5
C_LR=1e-4
G_LR=1e-4

USE_NOISE=True 
TRAIN_CURVE=True
TRAIN_F=True
FINETUNE_HEAD=False

PRETRAINED_F=None
PRETRAINED_G=None
L1_WEIGHT_MS=1e2
L1_WEIGHT_RGB=1e2
PERC_WEIGHT=1
REG_WEIGHT=0.1

conda activate array-doe

python train.py --result_path $RESULT_DIR --save_freq $SAVE_FREQ --n_epochs $EPOCH \
--patch_size $PATCH_SIZE --generator $GENERATOR --forward_model $FORWARD_MODEL --H_lr $H_LR --C_lr $C_LR --G_lr $G_LR \
--use_noise $USE_NOISE --train_curve $TRAIN_CURVE --train_F $TRAIN_F --finetune_head $FINETUNE_HEAD \
--pretrained_F $PRETRAINED_F --pretrained_G $PRETRAINED_G --l1_weight_ms $L1_WEIGHT_MS --l1_weight_rgb $L1_WEIGHT_RGB --perc_weight $PERC_WEIGHT --reg_weight $REG_WEIGHT
