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
import os, subprocess, shutil

#自动检测显存大小并动态设置 JT_SAVE_MEM 和限制
def _get_total_gpu_mem_mb() -> int | None:
    """Return total memory (MB) of the *first visible* GPU via nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    # honor CUDA_VISIBLE_DEVICES: pick the first visible id
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    index = 0
    if vis:
        try:
            index = int(vis.split(",")[0].strip())
        except Exception:
            index = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(index)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out.splitlines()[0])
    except Exception:
        return None

def _get_total_cpu_mem_mb() -> int | None:
    """Return total system RAM (MB) if psutil exists; else None."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        return None

def setup_jittor_memory_limits(
    gpu_frac: float = 0.9,   # cap GPU usage at 90% of total
    cpu_frac: float = 0.8,   # cap CPU RAM at 80% of total
    save_mem: bool = True,   # enable JT_SAVE_MEM
):
    # Enable Jittor "save memory" mode if requested
    if save_mem:
        os.environ.setdefault("JT_SAVE_MEM", "1")

    # GPU limit
    total_gpu_mb = _get_total_gpu_mem_mb()
    if total_gpu_mb and total_gpu_mb > 0:
        limit_mb = max(512, int(total_gpu_mb * gpu_frac))
        os.environ["device_mem_limit"] = str(limit_mb * 1024 * 1024)  # bytes

    # CPU limit
    total_cpu_mb = _get_total_cpu_mem_mb()
    if total_cpu_mb and total_cpu_mb > 0:
        limit_mb = max(1024, int(total_cpu_mb * cpu_frac))
        os.environ["cpu_mem_limit"] = str(limit_mb * 1024 * 1024)  # bytes

# call before importing jittor
setup_jittor_memory_limits()

import jittor as jt
from jittor import nn
import random
from jittor.dataset import DataLoader

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    cleanup
    # distributed_opt, 
    # distributed_gather, 
    # distributed_sync, 

)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

parser = argparse.ArgumentParser(description='jittor GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

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
         Imported from https://github.com/pyjt/examples/blob/master/imagenet/main.py#L247-L262
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
    # if args.fp16:
    #     with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
    #         _scaled_loss.backward()
    # else:
    #     _loss.backward()

    _optimizer.backward(_loss)

    if is_update:
        if args.clip > 0:
            _optimizer.clip_grad_norm(_model.parameters(), args.clip)

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
            # 保证是 Jittor Var 且 dtype 合适（索引: int32/64，mask: float32）
            _input  = data['input']  if isinstance(data['input'], jt.Var)  else jt.array(data['input'],  dtype=jt.int32)
            _target = data['target'] if isinstance(data['target'], jt.Var) else jt.array(data['target'], dtype=jt.int32)
            _msk    = data['mask']   if isinstance(data['mask'], jt.Var)   else jt.array(data['mask'],   dtype=jt.float32)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk)
            loss = _loss.mean()              # jt.Var
            avg_lm_loss.update(float(loss.item()))

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', float(loss.item()))

        # 可选：确保所有 kernel 执行完
        jt.sync_all(True)

    total_time = time.time() - start_time
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

    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)

        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        ) 

        _lm_loss = _lm_loss.mean() 

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(float(_lm_loss.data))
        optimizer_step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )

        if train_step % args.log_interval == 0: 
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if args.rank == 0: 
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()
        
        if train_step % args.save_interval == 0: 
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print('saving checkpoint', model_path)
                jt.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)
            # distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)

            model.train()
            # distributed_sync(args)

        if train_step == args.max_step:
            break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        jt.save({'model_state_dict': model.state_dict()}, model_path) 
    # distributed_sync(args)
    return train_step



if __name__ == '__main__':
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    jt.set_global_seed(args.random_seed)
    random.seed(args.random_seed)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )
 
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, drop_last=True,
    )
 
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, drop_last=False,
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(jt.load(args.init_checkpoint))    

    lm_net = lm_net.cuda()

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        #没有启动 mpirun 或 Jittor 多卡模式，实际上 world_size 应该是 1
        #如果保留原来的 args.world_size（是 0 或未初始化），就会导致 除以 0 或 max_step 计算错误
        args.world_size = 1
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    # lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                train_step=train_step, epoch=epoch
            )
  
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')


    # distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)