#!/bin/bash

python src/gpt2_ft.py \
  --train_data /root/autodl-tmp/data/e2e/sampled/train.jsonl \
  --valid_data /root/autodl-tmp/data/e2e/sampled/valid.jsonl \
  --train_batch_size 4 \
  --grad_acc 1 \
  --valid_batch_size 2 \
  --seq_len 512 \
  --model_card gpt2.md \
  --init_checkpoint /root/autodl-tmp/pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
  --platform local \
  --clip 0.0 \
  --lr 0.0002 \
  --weight_decay 0.01 \
  --correct_bias \
  --adam_beta2 0.999 \
  --scheduler linear \
  --warmup_step 500 \
  --max_epoch 5 \
  --save_interval 1000 \
  --lora_dim 4 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --label_smooth 0.1 \
  --work_dir /root/autodl-tmp/trained_models/GPT2_M/e2e_jittor \
  --random_seed 110
