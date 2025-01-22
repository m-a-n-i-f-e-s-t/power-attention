"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from power_attention import power_full
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-9 if input.dtype == torch.float32 else 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attention_kernel = config.attention_kernel
        self.block_size = config.block_size
        self.chunk_size = config.chunk_size
        self.degree = config.degree
        self.head_size = config.head_size
        self.qhead_ratio = config.qhead_ratio
        self.log_space = config.log_space
        self.gating = self.attention_kernel == 'power' and not config.disable_gating
        # key, query, value projections for all heads, but in a batch
        self.qkv_size = (config.qhead_ratio + 2) * self.n_head * self.head_size
        self.gating_size = config.n_head if self.gating else 0
        self.c_attn = nn.Linear(config.n_embd, self.qkv_size + self.gating_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.qhead_ratio * self.n_head * self.head_size, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if self.attention_kernel == 'power':
            self.ln = LayerNorm(self.head_size, bias=config.bias)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        h = self.n_head
        hq = self.qhead_ratio * h
        d = self.head_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkvg = self.c_attn(x)

        qkv = qkvg[...,:self.qkv_size]
        q, k, v  = qkv.split([hq*d, h*d, h*d], dim=2)
        q = q.view(B, T, hq, d)
        k = k.view(B, T, h, d)
        v = v.view(B, T, h, d)

        if self.gating:
            # 6.906768 corresponds to initial gating of .999
            log_g = torch.nn.functional.logsigmoid(6.906768 + qkvg[...,self.qkv_size:].to(dtype=torch.float32)).contiguous()
        else:
            log_g = None

        # apply rotary position embeddings
        r = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(2) # [1, T, 1]
        sincos = get_sinusoidal_embeddings(r, d, q.device)
        q = apply_rotary_position_embeddings(q, sincos)
        k = apply_rotary_position_embeddings(k, sincos)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.attention_kernel == 'sdpa':
            if self.gating: raise NotImplementedError('SDPA does not support gating')
            q = q.transpose(1, 2) # (B, nh, T, hs)
            k = k.transpose(1, 2) # (B, nh, T, hs)
            v = v.transpose(1, 2) # (B, nh, T, hs)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                 attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True,
                                                                 scale=1.0 / d**0.5,
                                                                 enable_gqa=self.qhead_ratio > 1)
            y = y.transpose(1, 2) # (B, T, nh, hs)
        elif self.attention_kernel == 'power':
            y = power_full(q.contiguous(), k.contiguous(), v.contiguous(), log_g,
                deg=self.degree,
                scale=1.0 / d**0.5,
                chunk_size=self.chunk_size)
            y = self.ln(y)
        else:
            msg = f'Unknown attention kernel: {self.attention_kernel}'
            raise NotImplementedError(msg)
        y = y.contiguous().view(B, T, hq * d) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_size = int(3.5 * config.n_embd) # Following Llama 3.1
        self.c_fc    = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def get_sinusoidal_embeddings(position, dim, device):
    """Generate sinusoidal positional embeddings."""
    # position is [B, T, nh]
    T = position.shape[1]
    div_term = (2. * math.pi) / (float(T) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)).view(1, 1, 1, -1)
    sinusoid_inp = position.unsqueeze(-1) * div_term
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    return sin, cos # [B, T, nh, d]

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.view(*x.shape[:-2], x.shape[-2] * x.shape[-1])

def apply_rotary_position_embeddings(x, sincos):
    _, T, _, _ = x.shape
    sin, cos = sincos
    sin = sin.repeat_interleave(2, dim=3)[:, :T]
    cos = cos.repeat_interleave(2, dim=3)[:, :T]
    return ((x * cos) + (rotate_every_two(x) * sin)).to(dtype=x.dtype)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention_kernel: str = 'sdpa'
    disable_gating: bool = False
    chunk_size: int = None
    degree: int = 2
    head_size: int = 64
    qhead_ratio: int = 1
    log_space: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))	

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            tail_idx = int(self.config.block_size*.1)
            tail_loss = F.cross_entropy(logits[:, -tail_idx:, :].reshape(-1, logits.size(-1)), targets[:, -tail_idx:].reshape(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = tail_loss = None

        return logits, loss, tail_loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx