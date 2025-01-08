## POWER FULL KERNEL ##
# Implements the power self-attention algorithm using CUDA kernels.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils._pytree import tree_map
from state_kernel._attention import attention, attention_reference
from state_kernel._attention.reference import attention_reference_fwd
from state_kernel._chunk_state import chunk_state, chunk_state_reference
from state_kernel._chunk_state.reference import chunk_state_reference_fwd
from state_kernel._discumsum import discumsum, discumsum_reference
from state_kernel._query_state import query_state, query_state_reference, ExpandedDim
from state_kernel._p1_full import p1_full
from state_kernel.utils import register_multi_grad_hook, clone_grads, get_statistics, plot_precision
from state_kernel.checks import clone_inputs
from state_kernel.timing_utils import report_fwd_bwd

from collections import defaultdict
import math
from collections import defaultdict
res = {
    'fwd': defaultdict(list),
    'ref_fwd': defaultdict(list),
    'bwd': defaultdict(list),
    'ref_bwd': defaultdict(list),
}


def dump_inputs(inputs, prefix, names):
    for name, input in zip(names, inputs):
        if isinstance(input, torch.Tensor):
            torch.save(input, f'{prefix}_{name}.pt')
            print(f'saved {name} to {prefix}_{name}.pt')
        else:
            print(f'{name} is not a tensor, value: {input}')

counter = 0
collect_interval = 1
total_samples = 20

def capture(Y, y, Y_test, y_test, Y_ref, y_ref, Y_gold, y_gold, ref_inputs, gold_inputs, test_inputs):
    if not Y.requires_grad:
        return
    Y_ref_error = (Y_ref - Y_gold).abs()
    y_ref_error = (y_ref - y_gold).abs()
    Y_test_error = (Y_test - Y_gold).abs()
    y_test_error = (y_test - y_gold).abs()
    res['ref_fwd']['Y'].append(get_statistics(Y_ref_error / Y_gold.abs()))
    res['ref_fwd']['y'].append(get_statistics(y_ref_error / y_gold.abs()))
    res['fwd']['Y'].append(get_statistics(Y_test_error / Y_gold.abs()))
    res['fwd']['y'].append(get_statistics(y_test_error / y_gold.abs()))


    def hook(grads):
        """ use Y_grad and y_grad as gradients for Y_ref, y_ref, Y_gold, and y_gold """
        global counter
        if counter % collect_interval != 0:
            counter += 1
        else:
            current_sample = len(res['ref_bwd'][0])
            print(f'collecting samples at {counter=}, {current_sample=}')
            counter += 1
            Y_grad, y_grad = grads
            torch.autograd.backward([Y_gold, y_gold], [Y_grad, y_grad])
            torch.autograd.backward([Y_ref, y_ref], [Y_grad, y_grad])
            torch.autograd.backward([Y_test, y_test], [Y_grad, y_grad])
            gold_inputs_grads = clone_grads(gold_inputs)
            ref_inputs_grads = clone_grads(ref_inputs)
            test_inputs_grads = clone_grads(test_inputs)
            for i in range(len(gold_inputs_grads)):
                ref_grad_error = (gold_inputs_grads[i] - ref_inputs_grads[i]).abs().to(torch.float32)
                test_grad_error = (gold_inputs_grads[i] - test_inputs_grads[i]).abs().to(torch.float32)
                res['ref_bwd'][i].append(get_statistics(ref_grad_error/gold_inputs_grads[i].abs()))
                res['bwd'][i].append(get_statistics(test_grad_error/gold_inputs_grads[i].abs()))
                if i == len(gold_inputs_grads) - 1 and len(res['ref_bwd'][i]) == total_samples:
                    dump_inputs(test_inputs, 'test', ['Q', 'S', 's', 'Y', 'y', 'rowmax', 'deg', 'stabilizer', 'zero_initial_state', 'ε', 'deterministic'])
                    dump_inputs(ref_inputs, 'ref', ['Q', 'S', 's', 'Y', 'y', 'rowmax', 'deg', 'stabilizer', 'zero_initial_state', 'ε', 'deterministic'])
                    print(f"collected {total_samples} samples, exiting")
                    ref_bwd_precisions = {key_idx: res['ref_bwd'][key] for key_idx, key in enumerate(res['ref_bwd'].keys())}
                    test_bwd_precisions = {key_idx: res['bwd'][key] for key_idx, key in enumerate(res['bwd'].keys())}
                    plot_precision(ref_bwd_precisions, test_bwd_precisions, range(0, total_samples * collect_interval, collect_interval), 'Iteration', 'Power Full BWD relative error')
                    ref_fwd_precisions = {key_idx: res['ref_fwd'][key] for key_idx, key in enumerate(res['ref_fwd'].keys())}
                    test_fwd_precisions = {key_idx: res['fwd'][key] for key_idx, key in enumerate(res['fwd'].keys())}
                    import pdb; pdb.set_trace()
                    plot_precision(ref_fwd_precisions, test_fwd_precisions, range(0, total_samples * collect_interval, collect_interval), 'Iteration', 'Power Full FWD relative error')

                    import sys
                    sys.exit()
            return Y_grad, y_grad

    
    register_multi_grad_hook([Y, y], hook)

DEBUG = False

res = {
    'fwd': defaultdict(list),
    'ref_fwd': defaultdict(list),
    'bwd': defaultdict(list),
    'ref_bwd': defaultdict(list),
}


def dump_inputs(inputs, prefix, names):
    for name, input in zip(names, inputs):
        if isinstance(input, torch.Tensor):
            torch.save(input, f'{prefix}_{name}.pt')
            print(f'saved {name} to {prefix}_{name}.pt')
        else:
            print(f'{name} is not a tensor, value: {input}')


def capture(Y, y, Y_test, y_test, Y_ref, y_ref, Y_gold, y_gold, ref_inputs, gold_inputs, test_inputs):
    if not Y.requires_grad:
        return
    Y_ref_error = (Y_ref - Y_gold).abs()
    y_ref_error = (y_ref - y_gold).abs()
    Y_test_error = (Y_test - Y_gold).abs()
    y_test_error = (y_test - y_gold).abs()
    res['ref_fwd']['Y'].append(get_statistics(Y_ref_error / Y_gold.abs()))
    res['ref_fwd']['y'].append(get_statistics(y_ref_error / y_gold.abs()))
    res['fwd']['Y'].append(get_statistics(Y_test_error / Y_gold.abs()))
    res['fwd']['y'].append(get_statistics(y_test_error / y_gold.abs()))


    def hook(grads):
        """ use Y_grad and y_grad as gradients for Y_ref, y_ref, Y_gold, and y_gold """
        Y_grad, y_grad = grads
        torch.autograd.backward([Y_gold, y_gold], [Y_grad, y_grad])
        torch.autograd.backward([Y_ref, y_ref], [Y_grad, y_grad])
        torch.autograd.backward([Y_test, y_test], [Y_grad, y_grad])
        gold_inputs_grads = clone_grads(gold_inputs)
        ref_inputs_grads = clone_grads(ref_inputs)
        test_inputs_grads = clone_grads(test_inputs)
        dump_inputs(test_inputs, 'test', ['Q', 'S', 's', 'Y', 'y', 'rowmax', 'deg', 'stabilizer', 'zero_initial_state', 'ε', 'deterministic'])
        dump_inputs(ref_inputs, 'ref', ['Q', 'S', 's', 'Y', 'y', 'rowmax', 'deg', 'stabilizer', 'zero_initial_state', 'ε', 'deterministic'])
        for i in range(len(gold_inputs_grads)):
            ref_grad_error = (gold_inputs_grads[i] - ref_inputs_grads[i]).abs()
            test_grad_error = (gold_inputs_grads[i] - test_inputs_grads[i]).abs()
            res['ref_bwd'][i].append(get_statistics(ref_grad_error/gold_inputs_grads[i].abs()))
            res['bwd'][i].append(get_statistics(test_grad_error/gold_inputs_grads[i].abs()))
            if i == len(gold_inputs_grads) - 1 and len(res['ref_bwd'][i]) == 20:
                print("collected 20 samples, exiting")
                ref_bwd_precisions = {key_idx: res['ref_bwd'][key] for key_idx, key in enumerate(res['ref_bwd'].keys())}
                test_bwd_precisions = {key_idx: res['bwd'][key] for key_idx, key in enumerate(res['bwd'].keys())}
                plot_precision(ref_bwd_precisions, test_bwd_precisions, range(20), 'Iteration', 'Power Full BWD relative error')
                ref_fwd_precisions = {key_idx: res['ref_fwd'][key] for key_idx, key in enumerate(res['ref_fwd'].keys())}
                test_fwd_precisions = {key_idx: res['fwd'][key] for key_idx, key in enumerate(res['fwd'].keys())}
                import pdb; pdb.set_trace()
                plot_precision(ref_fwd_precisions, test_fwd_precisions, range(20), 'Iteration', 'Power Full FWD relative error')

                import sys
                sys.exit()
        return Y_grad, y_grad

    
    register_multi_grad_hook([Y, y], hook)

DEBUG = False

def power_full(Q, K, V, log_G=None, initial_state=None,
               deg=2, stabilizer=None, ε=1e-5,
               chunk_size=None,
               deterministic=False,
               use_reference=False,
               normal_space=False): # noqa: C901
    """Compute power attention with optional gating and chunking.
    
    Implements the power attention algorithm with optional gating. On short sequences,
    this is equivalent to the quadratic power attention algorithm. On long sequences, we split 
    the sequence into chunks of size chunk_size (or chunk_n chunks) and process recurrently.

    Args:
        Q, K, V: torch.Tensor of shape [batch, seq, head, dim] - Query, Key and Value tensors
        log_G: Optional[torch.Tensor] of shape [batch, seq, head] - Optional log gating factors
        initial_state: Optional[Tuple[torch.Tensor]] - Not implemented, must be None
        deg: int - Power attention degree
        stabilizer: Optional[float] - Optional stabilization factor, defaults to expanded_dim for fp16
        ε: float - Small constant for numerical stability
        chunk_size: Optional[int] - Size of each chunk
        deterministic: bool - Whether to use deterministic gradient accumulation, this might slow things down with small batch sizes and require larger memory
        use_reference: bool - Whether to use reference implementation (not implemented)
        normal_space: bool - Whether to use normal space attention, this speeds up attention but at the risk of higher numerical instability

    Returns:
        torch.Tensor of shape [batch, seq, head, dim] - Output tensor

    Input restrictions:
        - Q, K, V must have same shape and dtype (32 or 64, and fp16 or bf16)
        - log_G must be float32 if provided
        - Must have at least 4 chunks
        - Sequence length must be evenly divisible by chunk count/size
    """
    # Defer to p1_full for deg=1
    if deg == 1:
        return p1_full(Q=Q, K=K, V=V, log_G=log_G, 
                       initial_state=initial_state, stabilizer=stabilizer, ε=ε, 
                       chunk_size=chunk_size, deterministic=deterministic,
                       use_reference=use_reference, normal_space=normal_space)
    
    # Swap in reference kernels if desired
    if use_reference:
        _attention = attention_reference
        _chunk_state = chunk_state_reference
        _discumsum = discumsum_reference
        _query_state = query_state_reference
    else:
        _attention = attention
        _chunk_state = chunk_state
        _discumsum = discumsum
        _query_state = query_state
    if initial_state is not None:
        raise NotImplementedError('Initial state not implemented')
    
    # Establish shapes and dtypes
    assert Q.dtype == K.dtype == V.dtype, 'dtypes of inputs must match'
    dtype = Q.dtype
    b, t, hq, d = Q.shape
    _, _, h, _ = K.shape
    assert hq % h == 0, f"Q heads must be a multiple of KV heads: {hq=} {h=}"
    qhead_ratio = hq // h
    if chunk_size is not None:
        c = chunk_size
        assert t % chunk_size == 0, f'{t=} not evenly divisible by {chunk_size=}'
        n = t // chunk_size
    else:
        c = t
        n = 1
    gating = log_G is not None
    if gating:
        log_G = log_G.to(torch.float32)

    if not stabilizer:
        stabilizer = 1.0

    # Quick return for simple quadratic attention
    if t <= c:
        if qhead_ratio > 1:
            K = K.repeat_interleave(qhead_ratio, dim=2)
            V = V.repeat_interleave(qhead_ratio, dim=2)
            if gating:
                log_G = log_G.repeat_interleave(qhead_ratio, dim=2)
        log_G_accum = log_G.cumsum(1) if log_G is not None else None
        Y, _, _ = _attention(Q, K, V, log_G_accum, deg, stabilizer, ε, deterministic, False, False, normal_space)
        assert Y.is_contiguous(), 'Y must be contiguous'
        return Y

    # Reshape into chunks
    Q = Q.view(b, n, c, hq, d)
    K = K.view(b, n, c, h, d)
    V = V.view(b, n, c, h, d)    
    if gating:
        log_G = log_G.view(b, n, c, h).to(torch.float32)
        log_G_intrachunk_accum = log_G.cumsum(2)

    # Compute chunk states
    if gating:
        log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
        cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
    else:
        cs_K = K
    S = _chunk_state(cs_K.contiguous(), V.contiguous(), deg)

    # Accumulate
    if gating:
        log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
    else:
        log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
    S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence
    S = S.narrow(1, 0, n)

    # Restrict to non-initial chunks for recurrent outputs
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    if gating:
        log_G_intrachunk_accum = log_G_intrachunk_accum.contiguous()

    # Compute attention
    Q_flatbatch = Q.view(b*n, c, hq, d)
    K_flatbatch = K.view(b*n, c, h, d)
    V_flatbatch = V.view(b*n, c, h, d)    
    log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum.view(b*n, c, h) if gating else None
    if qhead_ratio > 1:
        K_flatbatch = K_flatbatch.repeat_interleave(qhead_ratio, dim=2)
        V_flatbatch = V_flatbatch.repeat_interleave(qhead_ratio, dim=2)
        if gating:
            log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum_flatbatch.repeat_interleave(qhead_ratio, dim=2)
    attn_Y, _, rowmax = _attention(Q_flatbatch, K_flatbatch, V_flatbatch, log_G_intrachunk_accum_flatbatch, deg, stabilizer, ε, deterministic, False, False, normal_space)
    if normal_space: rowmax = 2 * torch.log(rowmax) # TODO(jbuckman): make the thing returned by rowmax consistent
    attn_Y = attn_Y.view(b, n, c, hq, d)
    rowmax = rowmax.view(b, n, c, hq).detach()

    if gating:
        if qhead_ratio > 1:
            log_G_intrachunk_accum = log_G_intrachunk_accum.repeat_interleave(qhead_ratio, dim=3)
        Q = Q * torch.exp(log_G_intrachunk_accum / deg).unsqueeze(-1).to(Q.dtype)
    if qhead_ratio > 1:
        S = S.repeat_interleave(qhead_ratio, dim=2)
    Y = _query_state(Q.contiguous(), S.contiguous(), attn_Y.contiguous(), (rowmax - torch.Tensor([math.log(stabilizer)]).to(rowmax.dtype).to(rowmax.device)).contiguous(), deg, 1.0, True, ε, deterministic)

    # Epilogue
    out = Y.contiguous().view(b, t, hq, d).to(dtype)
    return out

power_full_reference = partial(power_full, use_reference=True)

## Useful function to create sample inputs
def create_inputs(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, log_space=False, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h * qhead_ratio, d), dtype=dtype, device=device)
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device)
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device)
    if gating:
        log_G = F.logsigmoid(6.906768 + torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    initial_state = None
    stabilizer = 1./d**0.5
    ε = 1e-5
    normal_space = not log_space
    if requires_grad:
        Q, K, V, log_G, initial_state = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G, initial_state))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                initial_state=initial_state,
                deg=deg, stabilizer=stabilizer, ε=ε,
                chunk_size=chunk_size,
                normal_space=normal_space,
                deterministic=True)


## TUTORIAL ##
if __name__ == '__main__':
    # Create inputs
    t = 1024
    chunk_size=128
    b = 8
    h = 16
    d = 64
    deg = 2
    gating = True
    dtype = torch.float16
    inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)
    
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'profile':
        O = power_full(**inputs)
        torch.autograd.backward((O,), grad_tensors=(O,))
    else:
        # Benchmark
        print(f"Benchmarking power_full {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")

        report_fwd_bwd(power_full, **inputs)

