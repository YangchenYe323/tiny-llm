import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,              # [..., L_q, D_k]
    key: mx.array,                # [..., L_k, D_k]
    value: mx.array,              # [..., L_k, D_v]
    scale: float | None = None,
    mask: mx.array | None = None, # [..., L, L]
) -> mx.array:
    D_k = key.shape[-1]

    if scale is None:
        scale = 1.0 / mx.sqrt(D_k)

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
        wq: mx.array,     # [H*D_k, E]
        wk: mx.array,     # [H*D_k, E]
        wv: mx.array,     # [H*D_v, E]
        wo: mx.array,     # [E, H*D_v]
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


def causal_mask(L_q: int, L_v: int, dtype: mx.Dtype) -> mx.array:
    L = max(L_q, L_v)
    mask = mx.full([L, L], -mx.inf, dtype)
    mask = mx.triu(mask, 1)
    mask = mask[L-L_q:,L-L_v:]
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array, # [..., H_q, L_q, D]
    key: mx.array,   # [..., H_k, L, D]
    value: mx.array, # [..., H_v, L, D]
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    D = query.shape[-1]
    L_q = query.shape[-2]
    L_k = key.shape[-2]
    L_v = value.shape[-2]
    H_q = query.shape[-3]
    H_k = key.shape[-3]
    H_v = value.shape[-3]

    assert H_k == H_v
    assert H_q % H_k == 0

    n_repeat = H_q // H_k

    query_shape = query.shape
    # Expected reshape of the query. [..., H_k, n_repeat, L_q, D]
    query_reshape = list(query.shape[:-3]) + [H_k, n_repeat, L_q, D]
    query = mx.reshape(query, query_reshape)


    # Expected reshape of key. [..., H_k, 1, L_k, D]
    key_reshape = list(key.shape[:-3]) + [H_k, 1, L_k, D]
    key = mx.reshape(key, key_reshape)

    # Expected reshape of value. [..., H_k, 1, L_v, D]
    value_reshape = list(value.shape[:-3]) + [H_k, 1, L_v, D]
    value = mx.reshape(value, value_reshape)
    
    if scale is None:
        scale = 1.0 / mx.sqrt(D)

    # softmax(QK^t/d_k + mask)V
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        if isinstance(mask, mx.array):
            # Expected reshape of mask. [..., H_k, n_repeat, L_q, L_v]
            mask_reshape = list(mask.shape[:-3]) + [H_k, n_repeat, L_q, L_v]
            mask = mx.reshape(mask, mask_reshape)
        else:
            if mask == "causal":
                mask = causal_mask(L_q, L_v, query.dtype)
            else:
                raise Exception("Unknown mask type")
        scores = scores + mask

    out = mx.matmul(softmax(scores, axis=-1), value)
    out = mx.reshape(out, query_shape)
    return out


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
