#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
import jittor.distributions as dist

def add_gpu_params(parser: argparse.ArgumentParser):
    parser.add_argument("--platform", default='k8s', type=str, help='platform cloud')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank')
    parser.add_argument("--rank", default=0, type=int, help='rank')
    parser.add_argument("--device", default=0, type=int, help='device')
    parser.add_argument("--world_size", default=0, type=int, help='world size')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')

def parse_gpu(args):
    jt.set_global_seed(args.random_seed)
    jt.flags.use_cuda = 1
       
def cleanup(args):
    if args.platform == 'k8s' or args.platform == 'philly':
        args.dist.destroy_process_group()
        
#jittor可以自动单卡转化为MPI的多卡运行，删去了之前torch相关的分布式函数配置