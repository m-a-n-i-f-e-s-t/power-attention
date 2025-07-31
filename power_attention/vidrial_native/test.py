import pytest
import torch as th
from power_attention.vidrial_native.impl import PowerAttention, ChunkedPowerAttention
from vidrial.py_utils.test_utils import diff

import logging
logging.basicConfig(level=logging.DEBUG)

CASES = [(2, 128, 4, 32, 2, 16, 64),
        ]

@pytest.mark.parametrize("b,t,h,d,power,d_tile,chunk_size", CASES)
def test_run_AttentionAlgo(b, t, h, d, power, d_tile, chunk_size):
    assert t % chunk_size == 0
    dtype = th.float32
    algo = PowerAttention(b, t//chunk_size, chunk_size, h, d, power, d_tile, False, dtype)
    query, key, value = algo.make_query(), algo.make_key(), algo.make_value()

    output = PowerAttention.interface(query, key, value, power, d_tile, chunk_size)
    reference_output = PowerAttention.interface(query, key, value, power, d_tile, chunk_size, True)
    diff(output, reference_output, atol=1e-2)

@pytest.mark.parametrize("b,t,h,d,power,d_tile,chunk_size", CASES)
def test_run_ChunkedAttentionAlgo(b, t, h, d, power, d_tile, chunk_size):
    assert t % chunk_size == 0
    dtype = th.float32
    algo = ChunkedPowerAttention(b, t//chunk_size, chunk_size, h, d, power, d_tile, False, dtype)
    query, key, value = algo.make_query(), algo.make_key(), algo.make_value()

    chunked_output = ChunkedPowerAttention.interface(query, key, value, power, d_tile, chunk_size)
    chunked_reference_output = ChunkedPowerAttention.interface(query, key, value, power, d_tile, chunk_size, True)
    recurrent_output = ChunkedPowerAttention.interface(query, key, value, power, d_tile, 1)
    KVcache_output = ChunkedPowerAttention.interface(query, key, value, power, d_tile, t)

    diff(chunked_output, chunked_reference_output, atol=1e-2)
    diff(chunked_output, recurrent_output, atol=1e-2)
    diff(chunked_output, KVcache_output, atol=1e-2)
