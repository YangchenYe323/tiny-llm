import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = wk.shape[-2] // num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta

        print(f"hidden_size: {hidden_size}, num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {self.head_dim}") 
        print(f"wq: {wq.shape}, bq: {bq.shape}")
        print(f"wk: {wk.shape}, bk: {bk.shape}")
        print(f"wv: {wv.shape}, bv: {bv.shape}")
        print(f"wo: {wo.shape}")
        
        self.rope = RoPE(self.head_dim, max_seq_len, base=theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        q = linear(x, self.wq, self.bq)
        k = linear(x, self.wk, self.bk)
        v = linear(x, self.wv, self.bv)

        N, L_q, D_q = q.shape
        N, L_k, D_k = k.shape
        N, L_v, D_v = v.shape

        assert D_q == self.num_heads * self.head_dim
        assert D_k == self.num_kv_heads * self.head_dim
        assert D_v == self.num_kv_heads * self.head_dim

        q = mx.reshape(q, [N, L_q, self.num_heads, self.head_dim])
        k = mx.reshape(k, [N, L_k, self.num_kv_heads, self.head_dim])
        v = mx.reshape(v, [N, L_v, self.num_kv_heads, self.head_dim])
        q = self.rope(q)
        k = self.rope(k)

        # Transpose because attention expects array to be in shape
        # N, H, L, D
        # So each head can be processed in parallel
        q = mx.swapaxes(q, -2, -3)
        k = mx.swapaxes(k, -2, -3)
        v = mx.swapaxes(v, -2, -3)
        x = scaled_dot_product_attention_grouped(q, k, v, mask=mask)
        x = mx.swapaxes(x, -2, -3)

        x = mx.reshape(x, [N, L_q, D_q])
        x = linear(x, self.wo)

        return x



class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
