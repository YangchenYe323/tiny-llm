import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,              # [..., L, D]
    key: mx.array,                # [..., L, D]
    value: mx.array,              # [..., L, D]
    scale: float | None = None,
    mask: mx.array | None = None, # [..., L, L]
) -> mx.array:
    D = query.shape[-1]
    if scale is None:
        scale = 1.0 / mx.sqrt(D)

    # softmax(QK^t/d_k + mask)V
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask

    return mx.matmul(softmax(scores, axis=-1), value)



class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int, # H * D
        num_heads: int,   # H
        wq: mx.array,     # [H*D, E]
        wk: mx.array,     # [H*D, E]
        wv: mx.array,     # [H*D, E]
        wo: mx.array,     # [E, H*D]
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.head_dim = self.wq.shape[-1] // self.num_heads

    def __call__(
        self,
        query: mx.array,              # [..., L, E]
        key: mx.array,                # [..., L, E]
        value: mx.array,              # [..., L, E]
        mask: mx.array | None = None, # [..., L, L]
    ) -> mx.array:
        # [..., L, D]
        shape = query.shape
        # shape after projection, [..., L, H * D]
        proj_shape = list(shape[:-1]) + [self.num_heads * self.head_dim]
        # shape before transposing for multihead attention, [..., L, H, D]
        mh_shape = list(shape[:-1]) + [self.num_heads, self.head_dim] 
        # shape after transposing for multihead attention, [..., H, L, D]

        proj_qeury = linear(query, self.wq)
        proj_key = linear(key, self.wk)
        proj_value = linear(value, self.wv)

        proj_qeury = mx.reshape(proj_qeury, mh_shape)
        proj_key = mx.reshape(proj_key, mh_shape)
        proj_value = mx.reshape(proj_value, mh_shape)

        trans_query = mx.swapaxes(proj_qeury, -2, -3)
        trans_key = mx.swapaxes(proj_key, -2, -3)
        trans_value = mx.swapaxes(proj_value, -2, -3)

        scores = scaled_dot_product_attention_simple(trans_query, trans_key, trans_value, scale=None, mask=mask)

        trans_scores = mx.swapaxes(scores, -2, -3)
        trans_scores = mx.reshape(trans_scores, proj_shape)
        
        return mx.matmul(trans_scores, mx.swapaxes(self.wo, -2, -1))





def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
