#Inference

# 检查是否提供了检查点编号
if [ -z "$1" ]; then
  echo "请提供检查点编号，例如：./generate_e2e.sh 20000"
  exit 1
fi

checkpoint_number=$1

# 使用beam search生成输出
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/sampled/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/model.$checkpoint_number.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.$checkpoint_number.b10p08r4.jsonl