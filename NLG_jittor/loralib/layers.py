#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
#  Modified by Weihua, 20250703
#  -----------------------------------------------------------------------------------------
import jittor as jt
from jittor import nn, init

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.weight.new_zeros((r, num_embeddings))
            self.lora_B = self.weight.new_zeros((embedding_dim, r))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
            self.weight.stop_grad()
        self.reset_parameters()

    def reset_parameters(self):
        # nn.Embedding.reset_parameters(self)
        jittor_reset_parameters(self)
        init.gauss_(self.weight)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            init.zero_(self.lora_A)
            init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Embedding.train(self)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
    
    # Debug: Core difference between torch & Jittor. forward(Torch) <-> execute(Jittor)
    # Debug: torch.Tensor <-> jt.Var
    def execute(self, x: jt.Var):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.execute(self, x)
            # classjittor.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, dtype='float32')
            after_A = nn.embedding(
                x, self.lora_A.transpose(0, 1)
                # self.max_norm,self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.execute(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.weight.new_zeros((r, in_features))
            self.lora_B = self.weight.new_zeros((out_features, r))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
            self.weight.stop_grad()
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            init.zero_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def execute(self, x: jt.Var):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
            
        if self.r > 0 and not self.merged:
            result = nn.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            # self.lora_A = self.weight.new_zeros((r * sum(enable_lora), in_features))
            # self.lora_B = self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)) # weights for Conv1D with groups=sum(enable_lora)
            self.lora_A = self.weight.new_zeros((int(r * sum(enable_lora)), in_features))
            self.lora_B = self.weight.new_zeros((int(out_features // len(enable_lora) * sum(enable_lora)), r)) # weights for Conv1D with groups=sum(enable_lora)
            
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
            # Core: requires_grad = False(torch) <-> stop_grad() (Jittor)
            self.weight.stop_grad()
            # Compute the indices
            self.lora_ind = jt.zeros((out_features,), dtype="bool").view(len(enable_lora), -1)
            self.lora_ind[jt.array(enable_lora), :] = True
            
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            # self.weight.data = self.weight.data.transpose(0, 1)
            self.weight = self.weight.transpose(0, 1)

    def reset_parameters(self):
        # Debug: Jittor not support 'reset_parameters'
        # nn.Linear.reset_parameters(self)
        jittor_reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Debug: zeros_(torch), zero_(Jittor)
            init.zero_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        conv1d = nn.Conv1d(
            in_channels=self.lora_A.unsqueeze(0).shape[1],
            out_channels=self.lora_B.unsqueeze(-1).shape[0],
            kernel_size=self.lora_B.unsqueeze(-1).shape[2],
            groups=sum(self.enable_lora),
            bias=False
        )
        conv1d.weight.assign(self.lora_B.unsqueeze(-1))
        delta_w = conv1d(self.lora_A.unsqueeze(0)).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        nn.Linear.train(self)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def execute(self, x: jt.Var):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return nn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = nn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                # Error:    assert a.shape[-1] == b.shape[-1], (a.shape, b.shape)
                #           AssertionError: ([8,512,1024,], [1024,3072,])
                # Debug:  .T -> transpose(0, 1)
                result += self.lora_dropout(x) @ T(self.merge_AB().transpose(0, 1)) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            self.lora_B = self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.conv.weight.requires_grad = False
            self.conv.weight.stop_grad()
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        jittor_reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            init.zero_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train()
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def execute(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

# -----------------------------------------------------------------------------------------
# My Define Jittor
# -----------------------------------------------------------------------------------------
def jittor_reset_parameters(self):
    # 推荐使用 kaiming_uniform_ 做权重初始化
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    if self.bias is not None:
        fan_in, _ = jittor_calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

def jittor_calculate_fan_in_and_fan_out(w):
    # Jittor 实现 PyTorch 里 torch/nn/init.py
    dimensions = len(w.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = w.shape[1]
    num_output_fmaps = w.shape[0]
    receptive_field_size = 1

    if dimensions > 2:
        for s in w.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out