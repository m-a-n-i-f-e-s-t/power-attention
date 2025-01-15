"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import tiktoken

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
disable_eval = False
eval_interval = 100
eval_slowdown_factor = 1.2
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# logging
run_name = None
disable_logging = False
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
data_dir = os.path.expanduser('~/mai_datasets/ngpt_owt')
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
data_encoding = "bpe" # "bpe", "char", "bit"
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# manifest ai specific
attention_kernel = 'sdpa' # 'sdpa', 'power'
disable_gating = False
chunk_size = None
log_space = True
degree = 1
head_size = 64 # to separate from n_embd
qhead_ratio = 1
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# reencoding
bpe_enc = tiktoken.get_encoding('gpt2')
def reencode(seq):
    if data_encoding == "bpe":
        return seq
    seq_text = bpe_enc.decode(seq)
    char_enc_seq = [ord(c) for c in seq_text if ord(c) < 256]
    if len(char_enc_seq) < len(seq):
        char_enc_seq.extend([10] * (len(seq) - len(char_enc_seq))) # pad with newlines
    else:
        char_enc_seq = char_enc_seq[:len(seq)]
    if data_encoding == "char":
        return np.array(char_enc_seq, dtype=np.uint8)
    binary_enc_seq = [int(b) for n in char_enc_seq[:len(seq)//8+1] for b in format(n, '08b')][:len(seq)]
    if data_encoding == "bit":
        return np.array(binary_enc_seq, dtype=np.bool_)
    raise ValueError(f"Invalid data encoding: {data_encoding}")


# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - (block_size+1), (batch_size,))
    src = np.stack([reencode(data[i:i+block_size+1]) for i in ix]).astype(np.int64)
    x = torch.from_numpy(src[:, :-1])
    y = torch.from_numpy(src[:,1:  ])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  attention_kernel=attention_kernel, disable_gating=disable_gating, head_size=head_size,
                  chunk_size=chunk_size, degree=degree, log_space=log_space, qhead_ratio=qhead_ratio) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if data_encoding == "bpe":
        model_args['vocab_size'] = 50304
    elif data_encoding == "char":
        model_args['vocab_size'] = 256
    elif data_encoding == "bit":
        model_args['vocab_size'] = 2
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Invalid init_from: {init_from}")
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model...(takes a ~minute)", flush=True, end='')
    model = torch.compile(model) # requires PyTorch 2.0
    print("...done.")

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        tail_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, tail_loss = model(X, Y)
            losses[k] = loss.item()
            tail_losses[k] = tail_loss.item()
        out[f'{split}/avg'] = losses.mean()
        out[f'{split}/tail'] = tail_losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if master_process and not disable_logging:
    import logger
    run_name = logger.init(
        name=run_name,
        info={'config': config}
    )
    print(f"\033[37mLogging as \033[34m{run_name}\033[37m.\033[0m")
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
train_start_time = t0 = time.time()  # Start timing training section
total_eval_time = 0.0
raw_model = model.module if ddp else model # unwrap DDP container if needed
eval_count = 0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if master_process and not disable_eval and (iter_num == 0 or iter_num == max_iters or iter_num == int(eval_interval * (eval_slowdown_factor**(eval_count+1) - 1) / (eval_slowdown_factor - 1))):
        if torch.cuda.is_available(): torch.cuda.synchronize() # sync for timing
        eval_start_time = time.time()
        losses = estimate_loss()
        if torch.cuda.is_available(): torch.cuda.synchronize() # sync for timing
        total_eval_time += time.time() - eval_start_time
        eval_count += 1
        print(f"[{run_name}] step {iter_num} (eval {eval_count}): avg loss {losses['train/avg']:.4f}, val {losses['val/avg']:.4f} | tail loss train {losses['train/tail']:.4f}, val {losses['val/tail']:.4f}")
        if not disable_logging:
            logger.log("eval", {
                "iter": iter_num,
                "train.loss.avg": losses['train/avg'].item(),
                "heldout.loss.avg": losses['val/avg'].item(),
                "train.loss.tail": losses['train/tail'].item(),
                "heldout.loss.tail": losses['val/tail'].item(),
                "lr": lr,
                "train_hours": (time.time() - train_start_time - total_eval_time) / 3600,  # Convert seconds to hours
                "total_hours": (time.time() - train_start_time) / 3600,  # Convert seconds to hours
            })
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/avg_loss": losses['train/avg'],
                "val/avg_loss": losses['val/avg'],
                "train/tail_loss": losses['train/tail'],
                "val/tail_loss": losses['val/tail'],
                "lr": lr,
            })
        if losses['val/avg'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val/avg']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
        train_start = time.time()  # Resume training timer
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, _ = model(X, Y)
            # scale to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    dt = time.time() - t0
    t0 = time.time()
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if not disable_logging:
            logger.log("train", {
                "iter": iter_num,
                "train.loss": lossf,
                "lr": lr,
                "duration": dt,
                "train_hours": (time.time() - train_start_time - total_eval_time) / 3600,  # Convert seconds to hours
                "total_hours": (time.time() - train_start_time) / 3600,  # Convert seconds to hours
            })
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, runtime {(time.time() - train_start_time)/3600:.2f}h")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if not disable_logging:
    logger.wait_for_completion()
