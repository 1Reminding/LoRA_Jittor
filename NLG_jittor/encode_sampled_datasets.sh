#!/bin/bash

echo "Encoding sampled e2e datasets..."
path=data/e2e/sampled
echo "train..."
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted.jsonl --output $path/train.jsonl --add_bos --add_eos
echo "test..."
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted.jsonl --output $path/test.jsonl --add_bos --add_eos
echo "valid..."
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "Encoding sampled webnlg datasets..."
path=data/webnlg_challenge_2017/sampled
echo "train..."
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted.jsonl --output $path/train.jsonl --add_bos --add_eos
echo "test..."
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted.jsonl --output $path/test.jsonl --add_bos --add_eos
echo "valid..."
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "Encoding sampled dart datasets..."
path=data/dart/sampled
echo "train..."
python src/gpt2_encode.py --vocab vocab --input $path/train_formatted.jsonl --output $path/train.jsonl --add_bos --add_eos
echo "test..."
python src/gpt2_encode.py --vocab vocab --input $path/test_formatted.jsonl --output $path/test.jsonl --add_bos --add_eos
echo "valid..."
python src/gpt2_encode.py --vocab vocab --input $path/valid_formatted.jsonl --output $path/valid.jsonl --add_bos --add_eos

echo "Encoding complete!"