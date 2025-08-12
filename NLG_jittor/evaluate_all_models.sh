#!/bin/bash

# 定义模型检查点列表
# MODELS=("947" "1000" "1894" "2000" "2841" "3000" "3788" "4000" "4735")
MODELS=( "9465" )
# 为每个模型运行推理和评估
for model in "${MODELS[@]}"; do
  echo "Processing model.${model}.pt..."
  
  # 步骤1：生成输出 - 使用 Python 直接运行 Jittor 脚本
  python src/gpt2_beam.py \
      --data /root/autodl-tmp/data/e2e/sampled/test.jsonl \
      --batch_size 1 \
      --seq_len 512 \
      --eval_len 64 \
      --model_card gpt2.md \
      --init_checkpoint /root/autodl-tmp/trained_models/GPT2_M/e2e_jittor/model.${model}.pt \
      --platform local \
      --lora_dim 4 \
      --lora_alpha 32 \
      --beam 10 \
      --length_penalty 0.8 \
      --no_repeat_ngram_size 4 \
      --repetition_penalty 1.0 \
      --eos_token_id 628 \
      --work_dir ./trained_models/GPT2_M/e2e \
      --output_file predict.${model}.b10p08r4.jsonl
  
  # 步骤2：解码输出
  python src/gpt2_decode.py \
      --vocab ./vocab \
      --sample_file ./trained_models/GPT2_M/e2e/predict.${model}.b10p08r4.jsonl \
      --input_file /root/autodl-tmp/data/e2e/sampled/test_formatted.jsonl \
      --output_ref_file e2e_ref.${model}.txt \
      --output_pred_file e2e_pred.${model}.txt
  
  # 步骤3：评估结果 - 更新评估脚本路径
  echo "Evaluation results for model.${model}.pt:" > eval_results.${model}.txt
  python eval/e2e/measure_scores.py e2e_ref.${model}.txt e2e_pred.${model}.txt -p >> eval_results.${model}.txt
  
  echo "Completed evaluation for model.${model}.pt"
  echo "----------------------------------------"
done

echo "All evaluations completed!"