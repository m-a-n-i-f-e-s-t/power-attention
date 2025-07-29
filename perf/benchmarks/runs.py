from functools import wraps
import inspect
import torch
from flash_attn import flash_attn_func
from power_attention.triton import power_full
from power_attention.vidrial import power_full as power_full_vidrial
from power_attention.vidrial_fused import power_full as power_full_fused
from power_attention.create_inputs import create_inputs as create_inputs_power
from power_attention._attention import attention_triton, create_inputs as create_inputs_attention
from power_attention._attention.cuda import attention as attention_cuda
from power_attention._attention import create_inputs_cuda as create_inputs_attention_cuda
from power_attention._update_state import update_state_triton, update_state_vidrial, create_inputs as create_inputs_update_state
from power_attention._query_state import query_state_triton, query_state_vidrial, create_inputs as create_inputs_query_state
from power_attention._update_state.vidrial_fused import update_state as update_state_vidrial_fused
from power_attention._query_state.vidrial_fused import query_state as query_state_vidrial_fused
from power_attention._discumsum.triton import discumsum
from power_attention._discumsum import create_inputs as create_inputs_discumsum
from .flash import create_inputs as create_inputs_flash
from .sdpa import create_inputs as create_inputs_sdpa



def sanitize_kwargs(fn):
    """
    Sanitizes kwargs by removing any that are not in the function signature.
    """
    @wraps(fn)
    def wrapper(**kwargs):
        sig = inspect.signature(fn)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**valid_kwargs)
    return wrapper


class SDPA():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_sdpa)(**kw)
        return lambda: torch.nn.functional.scaled_dot_product_attention(**inputs)


class PowerFullTriton():
    @staticmethod
    def make_run(**kw):
        # 128 is not supported by power_full yet
        if kw['d'] == 128:
            def raise_not_implemented():
                raise NotImplementedError
            return raise_not_implemented

        inputs = sanitize_kwargs(create_inputs_power)(**kw)

        return lambda: power_full(**inputs)
        
    
class PowerFullVidrial():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_power)(**kw)
        return lambda: power_full_vidrial(**inputs)
    

class PowerFullFused():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_power)(**kw)
        return lambda: power_full_fused(**inputs)


class Discumsum():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_discumsum)(**kw)
        return lambda: discumsum(**inputs)

class QueryStateTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_query_state)(**kw)
        return lambda: query_state_triton(**inputs)


class UpdateStateTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_update_state)(**kw)
        return lambda: update_state_triton(**inputs)
    

class PowerAttentionCuda():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_attention_cuda)(**kw)
        def _run():
            o, _ = attention_cuda(**inputs)
            return o
        return _run

class PowerAttentionTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_attention)(**kw)
        def _run():
            o = attention_triton(**inputs)
            return o
        return _run


class UpdateStateVidrial():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_update_state)(**kw, use_vidrial_layout=True)
        return lambda: update_state_vidrial(**inputs)
    

class UpdateStateVidrialFused():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_update_state)(**kw, use_vidrial_layout=True)
        return lambda: update_state_vidrial_fused(**inputs)

class UpdateStateTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_update_state)(**kw)
        return lambda: update_state_triton(**inputs)


class QueryStateVidrial():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_query_state)(**kw, use_vidrial_layout=True)
        return lambda: query_state_vidrial(**inputs)


class QueryStateVidrialFused():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_query_state)(**kw, use_vidrial_layout=True, fused_norm=True)
        return lambda: query_state_vidrial_fused(**inputs)


class QueryStateTriton():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_query_state)(**kw)
        return lambda: query_state_triton(**inputs)
    

class FlashAttn():
    @staticmethod
    def make_run(**kw):
        inputs = sanitize_kwargs(create_inputs_flash)(**kw)
        return lambda: flash_attn_func(**inputs)