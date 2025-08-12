# LoRA_pytorch_reproduction & jittor_alignment

> **LoRA: Low-Rank Adaptation of Large Language Models** <br>
> *Edward J. Hu\*, Yelong Shen\*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* <br>
> Paper: https://arxiv.org/abs/2106.09685 <br>
> Github: https://github.com/microsoft/LoRA <br>

LoRA（Low-Rank Adaptation）是一种高效的预训练模型微调方法，旨在在保持模型性能的同时显著降低训练开销。  
<p>
<img src="img/lora.jpg" center="center" alt="lora" width="250" >
</p>

1. 实现方式  
- 在模型的部分权重矩阵（如注意力层的 $W_q$、$W_v$）引入低秩分解结构。  
- 将原权重 $W$ 表示为：  
  $$
  W' = W + BA
  $$
  其中 $A \in \mathbb{R}^{r\times k}$、$B \in \mathbb{R}^{d\times r}$，秩 $r \ll \min(d,k)$。  
- 冻结原有权重，仅训练新增的低秩矩阵 $A$、$B$。  

2. 优势  
- **参数高效**：大幅减少可训练参数量，通常降低至全量微调的 $0.1\%\sim1\%$。  
- **显存友好**：节省显存占用，支持更大批量或更长序列的训练。  
- **性能保持**：在多个下游任务上性能接近甚至匹配全量微调。  

## 复现任务要求
1. 使用pytorch和jittor两种框架对原论文进行复现，并进行性能对齐。

2. 将环境配置、数据准备脚本、训练脚本、测试脚本、与pytorch实现对齐的实验log，性能log都放在README中。

PS:如果计算资源有限，用少量数据的训练效果和pytorch版本的结果对齐.

* [X] [环境配置](#environment)
* [X] [数据准备脚本](#data-prepare)
* [X] [训练脚本](#seperate-steps)
* [X] [测试脚本](#seperate-steps)
* [X] [jittor与pytorch实现对齐的实验log](#log)
* [X] [性能Log](#log)

## Content

| Section                                 | Description     |
| --------------------------------------- | --------------- |
| [Overview](#overview)                      |                 |
| [Experiment](#experiment)                  | 环境配置，实验设计，注意事项  |
| [Train/Eval Log](#log--performance) | 日志、运行结果  |
| [Performance Comparison](#other-performance)    | 性能对比    |
| [Jittor Alignment](#jittor-alignment)        | Jittor 架构实现 |
| [Debug](#debug)                            | 常见问题及解决方案 |
| [Reference](#reference)                    |                 |
| [Citation](#citation)                      |                 |

## Overview


由于服务器资源有限，且原始LoRA仓库代码完整度较高，NLG实例上的三个数据集（e2e、webnlg、dart）复现方式除了e2e与webnlg、dart的评估需要切换评估脚本，但这个也是直接在原仓库中就写好的，其余部分没有实质性的区别。因此本仓库主要做了以下工作：
1. 基于Pytorch框架复现LoRA在 NLG 实例中e2e数据集上的所有实验，包括模型训练、推理、评估等，验证复现性能对齐原论文。(https://github.com/1Reminding/LoRA_Jittor/tree/main/NLG_pytorch)
2. 使用Jittor重写LoRA的torch实现，验证Jittor框架在NLG任务上的性能对齐。(https://github.com/1Reminding/LoRA_Jittor/tree/main/NLG_jittor)

本仓库的复现对齐原论文性能损失较小，且根据Jittor官方文档和相关资料重构代码，最终结果显示Jittor框架在NLG任务上的性能与Pytorch框架对齐。


## Experiment

### Environment

该仓库所有实验均在 AutoDL 平台租用云服务器实现，配置如下：
1. conda env1 (torch):
* 镜像 PyTorch 2.3.0 + Python 3.8(ubuntu22.04) + CUDA 12.1
* GPU RTX 3090(24GB) * 1
* CPU 14 vCPU Intel(R) Xeon(R) Gold 6330 
* 内存 90GB
* 硬盘 系统盘: 30 GB 数据盘: 50GB

具体环境中的 requirment 已导出所有包信息，torch(CUDA)版本在虚拟环境中设置为 1.7.1+cu101 （与原仓库相符）

2. conda env2 (jittor)
* jittor 1.3.10.0 + Python 3.9(ubuntu22.04) + CUDA 12.1

低于3.9 版本Python不兼容很多必要的包，具体环境中的 requirment 已导出所有包信息。

由于官方文档代码实现非常完整，复现过程中环境配置是保证代码运行的前提，下面是你在环境配置中可以参考的一些信息：

1. 安装 transformer 遇到 tokenizer rust编译的问题
<img src="img/tokenizer.jpg" alt="安装transformer遇到tokenizer rust编译的问题" width="400">
解决方案一（肯定能奏效）：

安装transformers 3.3.1但不安装其依赖，然后我们手动安装除tokenizers外的其他依赖。

```bash
pip install tokenizers==0.9.4
pip install transformers==3.3.1 --no-deps
pip install filelock regex sacremoses requests tqdm
```

不直接安装rust，因为用的是老的 transformers==3.3.1，它预期的 tokenizers 版本是 0.8.x。就算装了 Rust，pip 可能拉到较新的 tokenizers 源码（0.14/0.15…），与 3.3.1 API 不兼容，编出来也会冲突或运行时报错。

解决方案二（不一定奏效）：

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
pip install tokenizers==0.9.4
pip install transformers==3.3.1
```

直接安装rust，但是安装过程中可能会遇到一些问题，比如缺少一些库文件，这个时候需要根据错误提示去安装缺少的库文件。
2. 直接 pip install spacy 会出现的问题
- 需要 Cython 模块支持，需要先 pip install cython
- blis包的构建一直卡住，需要先 pip install blis==0.7.11
- version 'GLIBCXX_3.4.30' not found 错误
```bash
 ImportError: /root/miniconda3/envs/jittor/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found 
 ```
 Jittor 需要更新版本的 libstdc++.so.6 库，但你的 conda 环境中的版本较旧，不包含 GLIBCXX_3.4.30 符号.

 执行：
 ```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30
```
检查系统库是否包含所需版本。
如果系统 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 中已经包含 GLIBCXX_3.4.30 版本，创建一个软链接到 conda 环境中：
```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6
```
requirment中的包全部安装完成后，与requirment_pytorch 和requirment_jittor 进行对比，这是我当时冻结环境导出的完整包信息（或者现在就可以启动训练，根据报错进行必须包的安装）。
### Experiment Design

官方仓库包括两个实例：
1. NLG: 任务实现流程与原始 LoRA 论文中的 GPT-2 微调示例高度契合，代码依赖少，改动集中在模型与训练循环。(https://github.com/microsoft/LoRA/tree/main/examples/NLG)
2. NLU: 任务依赖 HuggingFace Transformers 中的高层封装与多任务接口，而 Jittor 在该部分生态支持不完善，实现难度和额外适配工作量更大。(https://github.com/microsoft/LoRA/tree/main/examples/NLU)

因此，本仓库进行 NLG 的复现，可快速验证 LoRA 的核心思想与性能，并便于与 PyTorch 版本进行对照测试。

### Config set
为保证性能损失在资源有限的情况下保持在合理范围内，且保证torch版本和jittor版本不会因为数据量和参数设置带来性能差异，需要进行如下配置：
1. 数据集：
缩小数据规模，但必须根据三个数据集中types的差异，均匀抽取样本，保证在train，test，val上均sampled 18% (保证在torch和jittor框架下不会爆显存，且不会超内存)。
2. 参数设置：
批次大小缩小为原论文的1/2，其他参数设置与原论文参数保持一致。
显存/内存占用降低：

batch size 和显存占用近似线性相关，减半 batch size 可以显著降低一次前向+反向的内存峰值，避免 OOM。每个 step 处理的样本减少，单步训练更快，适合显存、算力受限的环境。


### Start up

这里可能遇到的问题在前面的环境配置章节和后面的Debug章节中都有提到，也许有你需要的解决答案。

Clone repository

```bash
git clone https://github.com/1Reminding/LoRA_Jittor.git
cd LoRA_Jittor
```

Install dependencies 
注意torch和jittor最好新建两个虚拟环境，参见前面环境配置的版本要求。
Pytorch版本：

```bash
cd NLG_pytorch
pip install -r requirement.txt
```

Jittor版本：

```bash
cd NLG_jittor
pip install -r requirement.txt
```

**下面是NLG `e2e` 数据集整体实验流程：训练、推理、解码、评估，作为示例。对齐 torch 和 Jittor 性能。**

替换数据集，只需修改相应脚本中数据集的名称即可，将指令中的 `e2e` 替换为 `webnlg` 、`dart`

### Data prepare

数据预处理：

数据集在train，test，val上均 sampled 18%，且根据数据集的类型，区分数据中的不同types均匀采样(types : restaurant, hotel, attraction, train, bus, taxi...)

下面是对三个数据集中数据类型的部分展示：
```bash
E2E数据集按类别统计
==================================================
类别                  训练集      验证集      测试集      总计        
无类型                21950      1191       0          23141     
coffee shop          10396      3481       287        14164     
pub                  6531       0          2225       8756      
restaurant           3184       0          2181       5365      
总计                  42061      4672       4693       51426     

WebNLG数据集按类别统计(此处展示10个类别)
==================================================
类别                 训练集       验证集      测试集      总计        
Food                 1424       178        177        1779      
Airport              1090       136        136        1362      
Building             972        123        120        1215      
WrittenWork          937        118        116        1171      
SportsTeam           786        99         98         983       
Astronaut            530        67         66         663       
University           406        51         51         508       
City                 243        31         139        413       
ComicsCharacter      285        37         35         357       
Monument             267        32         33         332           
总计                  18025      2258       4928       25211      

DART数据集按类别统计 (此处展示20个类别)
==================================================
类别                 训练集      验证集      测试集       总计        
food                 9960       1063       1315       12338     
eattype              8196       1314       1793       11303     
pricerange           8151       839        1144       10134     
customer rating      7888       1039       772        9699      
area                 7139       1013       1433       9585      
familyfriendly       6492       839        1289       8620      
near                 5505       839        1819       8163      
country              2487       265        738        3490      
[title]              3162       61         69         3292      
location             1706       189        431        2326      
leader_name          1227       158        489        1874      
birth_place          438        48         788        1274      
club                 490        40         605        1135      
year                 1045       17         26         1088      
is_part_of           573        68         347        988       
language             608        56         247        911       
ingredient           652        86         168        906       
ethnic_group         543        57         293        893       
date                 855        22         15         892       
region               596        80         170        846       
总计                  62659      6980       12552      82191     
```
对数据集进行采样
```
python /root/LoRA/NLG_pytorch/sample_datasets.py
```
观察到终端输出如下信息，即可完成小规模数据集构建
```bash
处理 e2e 数据集...
处理 /root/LoRA/NLG_pytorch/data/e2e/train_formatted.jsonl...
  原始数据: 42061条, 抽样后: 7570条
处理 /root/LoRA/NLG_pytorch/data/e2e/valid_formatted.jsonl...
  原始数据: 4672条, 抽样后: 840条
处理 /root/LoRA/NLG_pytorch/data/e2e/test_formatted.jsonl...
  原始数据: 4693条, 抽样后: 844条

处理 webnlg 数据集...
处理 /root/LoRA/NLG_pytorch/data/webnlg_challenge_2017/train_formatted.jsonl...
  原始数据: 18025条, 抽样后: 3244条
处理 /root/LoRA/NLG_pytorch/data/webnlg_challenge_2017/valid_formatted.jsonl...
  原始数据: 2258条, 抽样后: 406条
处理 /root/LoRA/NLG_pytorch/data/webnlg_challenge_2017/test_formatted.jsonl...
  原始数据: 4928条, 抽样后: 886条

处理 dart 数据集...
处理 /root/LoRA/NLG_pytorch/data/dart/train_formatted.jsonl...
  原始数据: 62659条, 抽样后: 11278条
处理 /root/LoRA/NLG_pytorch/data/dart/valid_formatted.jsonl...
  原始数据: 6980条, 抽样后: 1256条
处理 /root/LoRA/NLG_pytorch/data/dart/test_formatted.jsonl...
  原始数据: 12552条, 抽样后: 2259条
  ```
数据条目总览：
|        | train           | test            | valid        |
| ------ | --------------- | --------------------- | ------------ |
| e2e    | 7570（42061）   | 844（4693）           | 840（4672）  |
| webnlg | 3244（18025）   | 886（4928）           | 406（2258）  |
| dart   | 11278（62659）  | 2259（12552）         | 1256（6980） |


### Run 
torch 版本和 jittor 版本只需要cd到两个对应的目录下运行即可（LoRA_Jittor\NLG_pytorch 和 LoRA_Jittor\NLG_jittor）
#### Train

```python
 # train_torch

python -m torch.distributed.launch --nproc_per_node=1 --use_env src/gpt2_ft.py \
    --train_data ./data/e2e/sampled/train.jsonl \
    --valid_data ./data/e2e/sampled/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
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
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```
jittor版本只需要修改上面脚本的第一句
```python
python src/gpt2_ft.py \
    --train_data ./data/e2e/sampled/train.jsonl \
    --valid_data ./data/e2e/sampled/valid.jsonl \
    ...
```
#### Inference

```python
 # 定义模型检查点列表
# MODELS=("1000","1839","2000","3000","3786","4000","5000","5679","6000","7000","7572","8000","9000","9465")
MODELS=( "9000" )
# 为每个模型运行推理和评估
for model in "${MODELS[@]}"; do
  echo "Processing model.${model}.pt..."

  # 步骤1：生成输出 - 使用 torchrun 替代 torch.distributed.launch
  torchrun --nproc_per_node=1 src/gpt2_beam.py \
      --data ./data/e2e/sampled/test.jsonl \
      --batch_size 1 \
      --seq_len 512 \
      --eval_len 64 \
      --model_card gpt2.md \
      --init_checkpoint ./trained_models/GPT2_M/e2e/model.${model}.pt \
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
```
对于jittor版本，仍然只需要修改第一句指令
```python
 time python src/gpt2_beam.py \
     --data ./data/dart/test_1k.jsonl \
     ...
```

#### Decode

```python
   # 步骤2：解码输出
   python src/gpt2_decode.py \
         --vocab ./vocab \
         --sample_file ./trained_models/GPT2_M/e2e/predict.${model}.b10p08r4.jsonl \
         --input_file ./data/e2e/sampled/test_formatted.jsonl \
         --output_ref_file e2e_ref.${model}.txt \
         --output_pred_file e2e_pred.${model}.txt
```

#### Evaluate

```python
   # 步骤3：评估结果 - 更新评估脚本路径
   echo "Evaluation results for model.${model}.pt:" > eval_results.${model}.txt
   python eval/e2e/measure_scores.py e2e_ref.${model}.txt e2e_pred.${model}.txt -p >> eval_results.${model}.txt
   
   echo "Completed evaluation for model.${model}.pt"
```

对于`webnlg` 和 `dart` 修改脚本使用 GenerationEval 进行评估。


ATTENTION: 原代码的逻辑已经在终端输出记录详细的 Log，注意保存

## Train/Eval Log

下面展示torch和jittor版本的性能，以及对齐性能log （完整文件位于两个框架文件夹的）。 

### dataset: e2e

截取部分训练过程记录的 log 展示如下，完整 log 记录查看 [log/replication-torch/e2e](log/replication-torch)：

```
====================================================================================================
Experiment dir : ./trained_models_jittor/GPT2_M/e2e
loading model pretrained weight.
set max_step: 5000
start to train the model................ 1
/root/LoRA/examples/NLG/src_jittor/gpt2_ft.py:214: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  avg_lm_loss.update(float(_lm_loss.data))
| epoch   1 step      100 |    100 batches | lr 4e-05 | ms/batch 544.36 | loss  4.56 | avg loss  5.60 | ppl 270.78
| epoch   1 step      200 |    200 batches | lr 8e-05 | ms/batch 541.69 | loss  3.26 | avg loss  3.80 | ppl 44.50
| epoch   1 step      300 |    300 batches | lr 0.00012 | ms/batch 542.20 | loss  2.94 | avg loss  3.16 | ppl 23.64
| epoch   1 step      400 |    400 batches | lr 0.00016 | ms/batch 542.96 | loss  2.51 | avg loss  2.97 | ppl 19.58
| epoch   1 step      500 |    500 batches | lr 0.0002 | ms/batch 543.07 | loss  3.73 | avg loss  2.93 | ppl 18.68
| epoch   1 step      600 |    600 batches | lr 0.000196 | ms/batch 544.30 | loss  2.74 | avg loss  2.88 | ppl 17.80
| epoch   1 step      700 |    700 batches | lr 0.000191 | ms/batch 542.84 | loss  3.09 | avg loss  2.87 | ppl 17.68
| epoch   1 step      800 |    800 batches | lr 0.000187 | ms/batch 543.30 | loss  2.95 | avg loss  2.84 | ppl 17.18
| epoch   1 step      900 |    900 batches | lr 0.000182 | ms/batch 543.63 | loss  3.29 | avg loss  2.88 | ppl 17.74
| epoch   1 step     1000 |   1000 batches | lr 0.000178 | ms/batch 543.98 | loss  2.82 | avg loss  2.77 | ppl 15.97
saving checkpoint ./trained_models_jittor/GPT2_M/e2e/model.1000.pt
eval samples: 0 loss: jt.Var([1.3142258], dtype=float32)
eval samples: 100 loss: jt.Var([1.2512419], dtype=float32)
average loss 1.4265571147203446
----------------------------------------------------------------------------------------------------
| Eval   1 at step     1000 | time: 25.58s | valid loss  1.43 | valid ppl  4.16 | best ppl  4.16 
----------------------------------------------------------------------------------------------------
saving checkpoint ./trained_models_jittor/GPT2_M/e2e/model.1000.pt
```

#### Alignment

分别展示 torch 和 Jittor 训练过程的 loss, avg_loss, valid_loss。可以观察到 500 步左右，基本收敛，训练loss最终保持在2.6左右。

<p>
<img src="figure/loss/loss_e2e.png" style="width:600; display: block; margin: 0 auto;">
</p>

将 torch, Jittor 的数据绘制在同一张图表，可以观察到avg_loss**基本重合**，说明实现了**性能对齐**，单步的loss出现不同的波动，属于正常现象。

<p>
<img src="figure/loss/all_loss_e2e.png" style="width:600; display: block; margin: 0 auto;">
</p>

#### Evaluation

运行评价指标函数，对齐性能，Evaluate 运算过程 log 如下：

```
Running MS-COCO evaluator...
creating index...
index created!
Loading and preparing results...   
DONE (t=0.00s)
creating index...
index created!
tokenization...
PTBTokenizer tokenized 8906 tokens at 88235.44 tokens per second.
PTBTokenizer tokenized 1065 tokens at 16000.19 tokens per second.
setting up scorers...
computing METEOR score...
METEOR: 0.471
computing Rouge score...
ROUGE_L: 0.741
computing CIDEr score...
CIDEr: 3.162
Running Py-MTEval metrics...
SCORES:
==============
BLEU: 0.6942
NIST: 8.0840
METEOR: 0.4708
ROUGE_L: 0.7408
CIDEr: 3.1612
```

注意，每次实验存在一定的误差，图表中最终展示的结果是多次实验取平均的结果。

观察到整体性能保持一致，Jittor的实验性能略好于torch，bias在2%左右。

<p align="center">
<img src="figure/sheet/d1-e2e.png" width=600>
</p>

绘制图表，更直观展示上述表格中的性能对比。

<p align="center">
<img src="figure/compare/d1-e2e.png" width=600>
</p>

### dataset2: webnlg

完整 log 记录查看 [log/replication-torch/webnlg](log/replication-torch/webnlg)：

#### Alignment

观察到 500 步左右，基本收敛，训练loss最终保持在2.0左右。

设定的是1000 step进行一次 eval，由于 webnlg 数据集本身数据量比较少，所以很早就结束训练。

<p>
<img src="figure/loss/loss_webnlg.png" style="width:600; display: block; margin: 0 auto;">
</p>

将 torch, Jittor 的数据绘制在同一张图表，可以观察到avg_loss**基本重合**，说明实现了**性能对齐**，单步的loss出现不同的波动，属于正常现象。

<p>
<img src="figure/loss/all_loss_webnlg.png" style="width:600; display: block; margin: 0 auto;">
</p>

#### Evaluation

运行评价指标函数，对齐性能。观察到整体性能保持一致，Jittor的实验性能略差于torch，bias在3%左右。

<p align="center">
<img src="figure/sheet/d2-webnlg.png" width=600>
</p>

绘制图表，更直观展示上述表格中的性能对比。

<p align="center">
<img src="figure/compare/d2-webnlg.png" width=600>
</p>

### dataset3: dart

完整 log 记录查看 [log/replication-torch/dart](log/replication-torch/webnlg)：

#### Alignment

观察到 2000 步左右，基本收敛，训练loss最终保持在2.7左右。

<p>
<img src="figure/loss/loss_dart.png" style="width:600; display: block; margin: 0 auto;">
</p>

将 torch, Jittor 的数据绘制在同一张图表，可以观察到avg_loss**基本重合**，说明实现了**性能对齐**，单步的loss出现不同的波动，属于正常现象。

<p>
<img src="figure/loss/all_loss_dart.png" style="width:600; display: block; margin: 0 auto;">
</p>

#### Evaluation

运行评价指标函数，对齐性能。观察到整体性能保持一致，Jittor的实验性能略差于torch，bias在3%左右。

<p align="center">
<img src="figure/sheet/d3-dart.png" width=600>
</p>

绘制图表，更直观展示上述表格中的性能对比。

<p align="center">
<img src="figure/compare/d3-dart.png" width=600>
</p>

### Summary

总结整体实验流程的记录，对齐 Jittor 与 torch 性能，保持训练参数完全一致。

综合多次实验取平均，train 过程 loss 下降趋势与随 step 的对应变化基本一致，整体的 evaluation 结果互有高低，bias 不超过5%。

综上，可以认为基本实现了 Jittor-torch 的性能对齐。

#### Alignment

<p align="center">
  <img src="figure/loss/all_loss_e2e.png" width="300" style="margin-right: 20px;">
  <img src="figure/loss/all_loss_webnlg.png" width="300" style="margin-right: 20px;">
  <img src="figure/loss/all_loss_dart.png" width="300">
</p>

#### Evaluation

<p align="center">
  <img src="figure/compare/d1-e2e.png" width="300" style="margin-right: 20px;">
  <img src="figure/compare/d2-webnlg.png" width="300" style="margin-right: 20px;">
  <img src="figure/compare/d3-dart.png" width="300">
</p>

## Other Performance

在主体实验训练、评估过程的记录之外，重点关注GPU显存占用，以及整体运行时间。

### GPU utilization

实验运行环境是单卡 RTX3090 24G，可以缩小数据规模后，按照官方仓库的参数配置复现实验。

`PROBLEM: 但是使用 Jittor 后同样参数运行，出现 OOM 的报错。`

<p align="center">
<img src="figure/sheet/gpu_utilization.png" width=600>
</p>


下面绘制图表，更直观展示上述表格内容。

<p align="center">
  <img src="figure/compare/gpu_utilization_rate.png" width="300" >
  <img src="figure/compare/pie.png" width="300">
</p>

综合收集的信息，切换不同的任务，不会对显存占用造成影响。整体实验参数的 batch_size 比较关键。

`IMPORTANT: 本仓库的实验观察，训练阶段(train) Jittor 显存占用高于 torch, 推理阶段(inference) Jittor 显存占用要低于 torch`

这就解释了原参数配置为什么 Jittor 会 OOM, torch 的实验已经基本达到 24G 的临界上限。Jittor 占用又高于 torch。

### Runtime

下述表格记录以分钟(min)为单位的训练、推理运行时间，出于简便，略去了秒的单位，但在log中可以找到详细的时间记录。

<p align="center">
<img src="figure/sheet/runtime.png" width="600" >
</p>

下面绘制图表，更直观展示上述表格内容。

<p align="center">
<img src="figure/compare/runtime.png" width="600">
</p>

直观观察到，在本仓库的复现实验的实际表现中，Jittor 的运行效率要低于 torch。

## Jittor Alignment‌

### 主要方法

1. 包文件导入，所有包含 torch 的地方直接替换成 Jittor
   ```
   # import torch
   # import torch.nn as nn
   import Jittor as Jt
   from Jittor import nn
   ```
2. tensor(torch) 替换成 Var(Jittor)
3. 模型结构中的forward(torch) 替换成 exexcute(Jittor)
4. 冻结参数，不计算梯度：**requires_grad=False** 替换成 **stop_grad()**
5. Other: 具体的函数接口替换，检索[6]官方的API文档。包括但不限于：
   * dataset, dataloader
   * 矩阵初始化
   * dtype
   * ...
   
主要参考：

- Jittor官方文档，包含对应torch的函数。
- ChatGPT，用于解决公开信息实现出错的问题。
- Jittor相关博客，包含Jittor的使用经验。
- GitHub相关仓库。

## Debug

### 1. Jittor 安装

```
sudo apt install libomp-dev
python -m pip install git+https://github.com/Jittor/jittor.git
```

运行 Jiitor 测试代码：

```
python -m jittor.test.test_example
```

`Error: ImportError: /root/miniconda3/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found <br>`
`Debug: Jittor 需要包含 GLIBCXX_3.4.30 符号版本的 C++ 标准库 (libstdc++.so.6)，当前 Conda 环境中提供的版本过旧，不包含这个符号。`

```
conda install -c conda-forge libstdcxx-ng -y
```

正常运行结果：

```
   step 990, loss = 0.0013174716150388122 {'hold_vars': 13, 'lived_vars': 61, 'lived_ops': 55}
   ... ...
   step 999, loss = 0.0009948192164301872 {'hold_vars': 13, 'lived_vars': 61, 'lived_ops': 55}
   ----------------------------------------------------------------------
   Ran 1 test in 14.363s
   OK
```

### 2. Jittor 加载模型权重文件

`ERROR:File "/root/LoRA/examples/NLG/src_jittor/gpt2_ft.py", line 405, in <module>`
`lm_net.load_weight(jt.load(args.init_checkpoint))`

Debug:

1. jt.load 内部调用了 safeunpickle，它尝试用 load_pytorch 加载 PyTorch 的 checkpoint。
2. load_pytorch 把 *.bin 文件当作一个 Zip 文件 来读（底层用 jt.ZipFile），因为 Jittor 的 PyTorch 兼容模块默认认为这是一个 .zip 格式的权重文件（类似 .pt / .pth 有时是 zip 存档）。

运行 `/NLG/model_process.py` 转换生成 xxx_zip.bin 进行训练。

```python
import torch
state_dict = torch.load('gpt2-medium-pytorch_model.bin', map_location='cpu')
torch.save(state_dict, 'gpt2-medium-pytorch_model_zip.bin', _use_new_zipfile_serialization=True)
```

### 3. 新版本 torch 参数兼容

`NLG/src/gpu.py` & `NLG/src_jittor/gpu.py`

local_rank 在新版本 torch 中弃用，补充参数处理兼容

```python
def add_gpu_params(parser: argparse.ArgumentParser):
    # parser.add_argument("--local_rank", default=0, type=int, help='local rank')
    # 修改一：参数传递
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', default=0, type=int,
                        help='local rank passed from distributed launcher.')
```

### 4. evaluation 中 meteor 函数计算错误

`NLG/eval/eval.py`

`ERROR:Error: test and reference not same length`

Debug: parse 函数中读取文件的方式。当使用 f.read().split('\n') 时，如果文件末尾有换行符，会产生一个额外的空字符串元素，导致列表长度不一。

替换成下述修改后的代码：

```python
# ... existing code ...

def meteor_score(references, hypothesis, num_refs, lng='en'):
    logging.info('STARTING TO COMPUTE METEOR...')
    print('STARTING TO COMPUTE METEOR...')
    hyps_tmp, refs_tmp = 'hypothesis_meteor', 'reference_meteor'

    # Filter out empty entries
    references_nonempty = []
    hypothesis_nonempty = []
    for i, refs in enumerate(references):
        if any(ref.strip() for ref in refs) and hypothesis[i].strip():
            references_nonempty.append(refs)
            hypothesis_nonempty.append(hypothesis[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis_nonempty)) 

    linear_references = []
    for refs in references_nonempty:
        for i in range(num_refs):
            linear_references.append(refs[i])

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    try:
# ... existing code ...
        result = subprocess.check_output(command, shell=True)
        meteor = result.split(b'\n')[-2].split()[-1]
    except:
# ... existing code ...
        print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        meteor = -1

    try:
# ... existing code ...

```

## Reference

1. LoRA official repo https://github.com/microsoft/LoRA
2. Jittor official repo https://github.com/Jittor/jittor
3. LoRA Jittor 1 https://github.com/waywooKwong/LoRA-Jittor


### Acknowledgement

在其他同学的复现代码中，我发现大量torch的代码，举个最简单的例子：
jt.save 的 safepickle 里会根据对象类型选择“如何保存”。当它发现像是“PyTorch 的对象/状态字典”（比如包含某些属性、或者模块路径上出现 torch 的特征），就会调用 save_pytorch，而这个实现里会 import torch。
类似的，其实有很多地方都并不是完全基于jittor框架，隐式或显式的调用了torch的代码，同一个环境中同时存在torch和jittor，我认为这可能会在其他应用时造成冲突，独立划分conda环境才是合理的选择。

## Citation

> 本仓库中的代码和实现思路欢迎借鉴和参考，用于学习、研究和复现。但请勿直接抄袭、原封不动复制粘贴本仓库的全部或部分代码。
>
> 如需引用或基于本仓库进行二次开发，请在显著位置注明来源并附上仓库链接：
>
> *This project is based on [1Reminding/LoRA_Jittor](https://github.com/1Reminding/LoRA_Jittor).*

```BibTeX
@inproceedings{
hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
```

