#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import random
import jittor as jt
from jittor import nn

from model import GPT2LMModel, GPT2Config  
from data import FT_Dataset          # 替代 DataLoader 的数据集封装逻辑
from exp_utils import create_exp_dir
import loralib as lora

parser = argparse.ArgumentParser(description='Jittor GPT2 ft script')

# add_gpu_params(parser)
# add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    # 做梯度累积缩放：等价于对大 batch 的平均 loss 反传
    _loss = _loss / max(1, args.grad_acc)
    _optimizer.backward(_loss)

    if is_update:
        if args.clip > 0:
            # 用 Jittor 的裁剪接口
            nn.clip_grad_norm(_model.parameters(), args.clip)

        _optimizer.step()
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()

def evaluate(model, valid_loader, args):
    model.eval()
    start_time = time.time()
    avg_lm_loss = AverageMeter()

    with jt.no_grad():
        for idx, data in enumerate(valid_loader):
            # 保证 batch 中是 jt.Var
            batch = {k: (v if isinstance(v, jt.Var) else jt.array(v))
                     for k, v in data.items()}
                # 2) 规范 dtype（关键就这三行）
            batch['input']  = batch['input'].int32()     # token ids
            batch['target'] = batch['target'].int32()    # labels
            batch['mask']   = batch['mask'].float32()    # loss mask / attention mask   

            lm_logits, loss = model(
                batch['input'],
                lm_labels=batch['target'],
                lm_mask=batch['mask']
            )
            loss = loss.mean()
            loss.sync()                     # 只同步这一个标量即可

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', f'{loss.item():.6f}')

    print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

def train_validate(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    args,
    train_step=0,
    epoch=0
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    # PyTorch DDP 专用，Jittor 无需：
    # train_loader.sampler.set_epoch(epoch)

    for idx, data in enumerate(train_loader):
        # 1) 转成 jt.Var
        batch = {k: (v if isinstance(v, jt.Var) else jt.array(v)) for k, v in data.items()}
         # 2) 规范 dtype（关键就这三行）
        batch['input']  = batch['input'].int32()
        batch['target'] = batch['target'].int32()
        batch['mask']   = batch['mask'].float32()

        _lm_logits, _lm_loss = model(
            batch['input'],
            lm_labels=batch['target'],
            lm_mask=batch['mask'],
            label_smooth=args.label_smooth
        )
        _lm_loss = _lm_loss.mean()
        _lm_loss.sync()  # 同步，便于 .item()

        train_step += 1
        is_update = (train_step % args.grad_acc == 0)
        avg_lm_loss.update(_lm_loss.item())

        # 这里内部会做 loss/(grad_acc) 的处理（或你也可在外面先除）
        optimizer_step(_lm_loss, optimizer, model, scheduler, args, is_update=is_update)

        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            # 兼容两种取 lr 的方式
            try:
                lr = optimizer.param_groups[0]['lr']
            except Exception:
                lr = getattr(optimizer, "lr", None)
            log_str = (
                f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | '
                f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | '
                f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | '
                f'ppl {math.exp(avg_lm_loss.avg):5.2f}'
            )
            print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

        # save checkpoint
        if train_step % args.save_interval == 0:
            os.makedirs(args.work_dir, exist_ok=True)
            model_path = os.path.join(args.work_dir, f'model.{train_step}.pkl')
            print('saving checkpoint', model_path)
            # 如果你有 jittor 版 lora 工具，可优先保存 lora 权重
            try:
                import loralib_jt as lora_jt
                state = lora_jt.lora_state_dict(model)
            except Exception:
                state = model.state_dict()
            jt.save(state, model_path)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            log_str = (
                f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | '
                f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | '
                f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '
            )
            print('-' * 100)
            print(log_str)
            print('-' * 100)

            model.train()

        if args.max_step is not None and train_step >= args.max_step:
            break

    # final save
    os.makedirs(args.work_dir, exist_ok=True)
    model_path = os.path.join(args.work_dir, f'model.{train_step}.pkl')
    print('saving checkpoint', model_path)
    jt.save(model.state_dict(), model_path)

    return train_step

if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args)

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.random_seed)
    random.seed(args.random_seed)

    # 单卡：固定 rank=0
    args.rank = 0
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    # ===== 数据 =====
    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len,
        joint_lm=(args.obj=='jlm')
    )
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len
    )

    # 極簡 Loader：假設 FT_Dataset 的 __getitem__ 就返回“已分好批”的 dict
    class SimpleLoader:
        def __init__(self, dataset, shuffle=False):
            self.dataset, self.shuffle = dataset, shuffle
        def __len__(self): return len(self.dataset)
        def __iter__(self):
            idxs = list(range(len(self.dataset)))
            if self.shuffle: random.shuffle(idxs)
            for i in idxs:
                yield self.dataset[i]

    train_loader = SimpleLoader(train_data, shuffle=True)
    valid_loader = SimpleLoader(valid_data, shuffle=False)

    # ===== 模型配置 =====
    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12,
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16,
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        )
    else:  # gpt2.lg
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20,
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)

    # ===== 预训练权重加载 =====
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        try:
            state = jt.load(args.init_checkpoint)
            # 如果你的 Jittor 模型实现了同名接口：
            if hasattr(lm_net, "load_weight"):
                lm_net.load_weight(state)
            else:
                lm_net.load(args.init_checkpoint)   # 或者直接用你实现的 model.load
        except Exception as e:
            print(f"[WARN] load failed via jt.load: {e}; trying model.load(path)")
            lm_net.load(args.init_checkpoint)

    # 不需要 .cuda()（Jittor 用全局 flag 控制）
    # lm_net = lm_net.cuda()

    # LoRA（若你有 Jittor 版 lora 工具）
    
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)

    # ===== 优化器（用 Jittor 自带）=====
    optimizer = nn.Adam(lm_net.parameters(), lr=args.lr, beta2=args.adam_beta2, weight_decay=args.weight_decay)

    # ===== 训练步数 =====
    if args.max_step is None:
        args.max_step = args.max_epoch * len(train_loader)
        print('set max_step:', args.max_step)

    # ===== 简易线性 scheduler（warmup+decay）=====
    class LinearScheduler:
        def __init__(self, opt, warmup_steps, total_steps):
            self.opt = opt
            self.warmup = warmup_steps
            self.total = max(total_steps, 1)
            self.step_num = 0
            self.base_lr = [g['lr'] for g in self.opt.param_groups]
        def step(self):
            self.step_num += 1
            if self.warmup > 0 and self.step_num < self.warmup:
                scale = self.step_num / self.warmup
            else:
                scale = max(0.0, 1.0 - (self.step_num - self.warmup) / max(1, self.total - self.warmup))
            for pg, lr0 in zip(self.opt.param_groups, self.base_lr):
                pg['lr'] = lr0 * scale

    scheduler = LinearScheduler(optimizer, warmup_steps=getattr(args, 'warmup_step', 0), total_steps=args.max_step)

    # AMP（如需）：Jittor 方式（可选）
    # if args.fp16:
    #     jt.flags.amp_level = 1  # 或 jt.flags.use_fp16 = 1，视本地版本

    # 训练循环
    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args,
                train_step=train_step, epoch=epoch
            )
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                print('-' * 100)
                print('End of training')
                break
    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')

    # 分布式清理（Jittor 单卡不需要）
    # cleanup(args)
    print('Done.')