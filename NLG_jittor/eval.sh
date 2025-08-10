#!/bin/bash

model=9465

echo "正在评估 model.${model}.pt"
echo "Evaluation results for model.${model}.pt:" > eval_results.${model}.txt
python eval/e2e/measure_scores.py e2e_ref.${model}.txt e2e_pred.${model}.txt -p >> eval_results.${model}.txt

echo "Completed evaluation for model.${model}.pt"
echo "---------------------------------------------"
