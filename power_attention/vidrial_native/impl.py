from dataclasses import dataclass
from math import sqrt
from functools import lru_cache
import torch as th
from vidrial.kernels.sympow.dimensions import sympow_dim
import vidrial.kernels.sympow_mma.interface as ops
from power_attention._attention.triton import attention

def default_d_tile(d, power):
    default_d_tiles = {1: 16, 2: 8, 3: 4, 4: 2}
    d_tile = None
    if power in default_d_tiles.keys():
        d_tile = default_d_tiles[power]
        d_tile = d_tile if d % d_tile == 0 else 1
    return d_tile or 1
def default_chunk_size(t, d, power):
    assert False, "not implemented"

@dataclass
class PowerAttention:
    """ Applies the attention algorithm within chunks
         O_i = sum_{j=0}^{c} (Q_i^T K_j)^p V_j
    Dimensions:
        c: is the chunk time dimension
        d: is the feature dimension
        b, n, h: are all batch dimensions
    """
    b: int # batch dimension
    n: int # chunk number along time dimension
    c: int # chunk along the time dimension
    h: int # head dimension
    d: int # feature dimension
    power: int
    d_tile: int
    use_reference: bool
    dtype: th.dtype

    @property
    def D(self): return sympow_dim(self.d, self.power, self.d_tile)
    @property
    def t(self): return self.n * self.c

    @property
    def query_shape(self): return (self.b, self.t, self.h, self.d)
    @property
    def key_shape(self): return (self.b, self.t, self.h, self.d)
    @property
    def value_shape(self): return (self.b, self.t, self.h, self.d)
    @property
    def output_shape(self): return (self.b, self.t, self.h, self.d)

    def make_query(self): return th.randn(self.query_shape, dtype=self.dtype, device='cuda') / sqrt(self.d)
    def make_key(self): return th.randn(self.key_shape, dtype=self.dtype, device='cuda') / sqrt(self.d)
    def make_value(self): return th.randn(self.value_shape, dtype=self.dtype, device='cuda') / sqrt(self.d)

    def make_inputs(self, requires_grad=False):
        tensors = self.make_query(), self.make_key(), self.make_value()
        for t in tensors: t.requires_grad = requires_grad
        return {'query': tensors[0],
                'key': tensors[1],
                'value': tensors[2],
                'power': self.power,
                'd_tile': self.d_tile,
                'chunk_size': self.c,
                'use_reference': self.use_reference}

    @classmethod
    def interface(cls, query, key, value, power, d_tile=None, chunk_size=None, use_reference=False):
        """ User interface for power attention
        Args:
            query: Tensor of shape [b, t, h, d]
            key:   Tensor of shape [b, t, h, d]
            value: Tensor of shape [b, t, h, d]
            power: int
            d_tile: int | None, if none the default d_tile for each power is used
            chunk_size: int | None, if none the default chunk_size for each power is used
        Returns:
            O: Tensor of shape [b, t, h, d]
        """
        b, t, h, d = query.shape
        assert query.shape == key.shape == value.shape
        assert query.dtype == key.dtype == value.dtype
        dtype = query.dtype
        d_tile = d_tile or default_d_tile(d, power)
        chunk_size = chunk_size or default_chunk_size(t, d, power)
        assert t % chunk_size == 0
        self = cls(b, t//chunk_size, chunk_size, h, d, power, d_tile, use_reference, dtype)
        Q, K, V = map(lambda x, shape: x.reshape(shape), (query, key, value), (self.Q_shape, self.K_shape, self.V_shape))
        O = self.algorithm(Q, K, V)
        return O.reshape(self.output_shape)

    @property
    def Q_shape(self): return (self.b, self.n, self.c, self.h, self.d)
    @property
    def K_shape(self): return (self.b, self.n, self.c, self.h, self.d)
    @property
    def V_shape(self): return (self.b, self.n, self.c, self.h, self.d)
    @property
    def O_shape(self): return (self.b, self.n, self.c, self.h, self.d)

    def reference_algorithm(self, Q, K, V):
        assert (Q.shape, K.shape, V.shape) == (self.Q_shape, self.K_shape, self.V_shape)
        Q, K, V = map(lambda A: A.transpose(2, 3), (Q, K, V)) # transpose c and h
        M = th.tril(th.ones((self.c, self.c), dtype=th.bool, device='cuda'))
        A = (Q @ K.transpose(-1, -2)) * M[None, None, :, :]
        O = A**self.power @ V
        O = O.transpose(2, 3) # transpose c and h
        assert O.shape == self.O_shape
        return O

    def algorithm(self, Q, K, V):
        call_triton = not self.use_reference
        call_triton = call_triton and self.c % 64 == 0 # triton kernel assumes chunk size is a multiple of 64
        if call_triton:
            _Q, _K, _V = map(lambda A: A.reshape(self.b*self.n, self.c, self.h, self.d), (Q, K, V))
            _O, l, rowmax = attention(_Q, _K, _V, None, self.power, norm=False)
            _O =  _O * th.exp(rowmax.detach())[..., None] # Undo the rowmax correction
            O = _O.reshape(*self.O_shape)
            return O
        return self.reference_algorithm(Q, K, V)


@dataclass
class ChunkedPowerAttention(PowerAttention):
    """ The Chunked Power Attention Algorithm
    By splitting the time timension into chunks t = n * c we gain a mechanism to
    interpolate between the recurrent O(tdD)) and KVcache O(dt^2) algorithms
    In the case wehre n = 1 or c = 1 we skip unnecessary computations
    """

    @property
    def sympowA_mm(self): return ops.sympowA_mm_reference if self.use_reference else ops.sympowA_mm
    @property
    def sympowA_mma_(self): return ops.sympowA_mma_reference_ if self.use_reference else ops.sympowA_mma_

    @property
    def S_shape(self): return (self.b, self.n-1, self.h, self.D, self.d)

    def algorithm(self, Q, K, V):
        assert (Q.shape, K.shape, V.shape) == (self.Q_shape, self.K_shape, self.V_shape)
        O = PowerAttention.algorithm(self, Q, K, V)
        if self.n > 1:
            # transpose c and h and remove irrelevant chunk
            _K, _V = map(lambda M: M.transpose(2, 3)[:,:-1], (K, V))
            _Q = Q.transpose(2, 3)[:,1:]
            # Compute the state at the beginning of each chunk
            S = self.sympowA_mm(_K.transpose(-1, -2), _V, dim=-2, power=self.power, d_tile=self.d_tile)
            S = th.cumsum(S, dim=1)
            assert S.shape == self.S_shape
            # Query the initial states of each chunk
            _O = self.sympowA_mm(_Q, S, dim=-1, power=self.power, d_tile=self.d_tile)
            O = O.clone()
            O[:,1:] += _O.transpose(2, 3)
            # TODO: get inplace working
            # _Q, _O = map(lambda M: M.transpose(2, 3)[:,1:], (Q, O))
            # self.sympowA_mma_(_Q, S, _O, dim=-1, power=self.power, d_tile=self.d_tile)
        return O


# @dataclass
# class ChunkedPowerAttentionNormalized(ChunkedPowerAttention):

#     @property
#     def s_shape(self): return (self.b, self.n-1, self.h, self.D, 1)

#     def algorithm(self, Q, K, V):
#         """ The chunked algorithm for power attention
#         Args:
#             Q: Tensor of shape [b, n, c, h, d]
#             K: Tensor of shape [b, n, c, h, d]
#             V: Tensor of shape [b, n, c, h, d]
#         Returns:
#             O: Tensor of shape [b, n, c, h, d]
#         """
#         assert (Q.shape, K.shape, V.shape) == (self.Q_shape, self.K_shape, self.V_shape)
#         V[..., 0] = 1. # First feature is reserved for normalization
#         Oattn = PowerAttention.algorithm(self, Q, K, V)
#         if self.n > 1:
#             # transpose c and h and remove irrelevant chunk
#             _K, _V = map(lambda M: M.transpose(2, 3)[:,:-1], (K, V))
#             _Q, _Oattn = map(lambda M: M.transpose(2, 3)[:,1:], (Q, Oattn))
#             # Compute the state at the beginning of each chunk
#             S = self.sympowA_gemm(1/sqrt(self.D), _K.transpose(-1, -2), _V, None, None, dim=-2, power=self.power, d_tile=self.d_tile)
#             S = th.cumsum(S, dim=1)
#             assert S.shape == self.S_shape
#             # Compute the 
#             s = S[..., 1:]
#             assert s.shape == self.s_shape
#             ostate = self.sympowA_mm(Q, s, power=self.power, dim=-1)
#             # Query the initial states of each chunk
#             new_o = oattn + ostate
#             self.sympowA_gemm(new_o, Q, S, new_o/oattn, O, power=self.power, dim=-1)
#         return O

    

# def power_attention(query, key, value, power, log_discount=None, normalize=True, d_tile=None, chunk_size=None):
#     """ User interface for power attention
#     Args:
#         query: Tensor of shape [b, t, h, d]
#         key:   Tensor of shape [b, t, h, d]
#         value: Tensor of shape [b, t, h, d]
#         power: int
#         log_discount: None | Tensor of shape [b, t, h], if none the default log_discount for each power is used
#         normalize: bool, if true applies temporal normalization to the outputs
#         d_tile: int | None, tile size for the sympow expansion if None selects a reasonable default
#         chunk_size: int | None, if None the default chunk_size for each power is used
#     Returns:
#         O: Tensor of shape [b, t, h, d]
#     """
#     assert log_discount is None, "discounting is not implemented"
#     f = ChunkedPowerAttentionNormalized.interface if normalize else ChunkedPowerAttention.interface
#     return f(query, key, value, power, d_tile, chunk_size)
