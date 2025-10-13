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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        """
        Qwen2 MLP block:

        MLP(X) = W_down(SiLU(W_gate(X)) * W_up(X))

        Dimensions:
            x: [..., L, E]
            w_gate: [I, E]
            w_up: [I, E]
            w_down: [E, I]
        Returns:
            [..., L, E]
        """

        return linear(
            silu(linear(x, self.w_gate))
            *
            linear(x, self.w_up),
            self.w_down
        )


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
        self.attention = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta
        )

        D = wq.shape[-1]

        self.input_layernorm = RMSNorm(
            dim=D,
            weight=w_input_layernorm,
            eps=rms_norm_eps
        )

        self.post_attention_layernorm = RMSNorm(
            dim=D,
            weight=w_post_attention_layernorm,
            eps=rms_norm_eps
        )

        self.mlp = Qwen2MLP(
            dim=D,
            hidden_dim=hidden_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        attn = x + self.attention(self.input_layernorm(x), mask=mask)
        return attn + self.mlp(self.post_attention_layernorm(attn))


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        print(mlx_model.args)
        vocab_size = mlx_model.args.vocab_size
        hidden_size = mlx_model.args.hidden_size
        # num_hidden_layers = mlx_model.args.num_hidden_layers
        intermediate_size = mlx_model.args.intermediate_size
        num_attention_heads = mlx_model.args.num_attention_heads
        rms_norm_eps = mlx_model.args.rms_norm_eps
        num_kv_heads = mlx_model.args.num_key_value_heads
        max_position_embeddings = mlx_model.args.max_position_embeddings
        theta = mlx_model.args.rope_theta
        # traditional = mlx_model.args.rope_traditional
        # rope_scaling = mlx_model.args.rope_scaling

        # If true, use the embedding layer as the last linear layer,
        # otherwise use a linear layer defined by model.lm_heads
        tie_word_embeddings = mlx_model.args.tie_word_embeddings

        # extract and dequantize embedding block
        embedding_weight = dequantize_linear(mlx_model.model.embed_tokens)
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=hidden_size,
            weight=embedding_weight,
        )

        self.layers = []
        for layer in mlx_model.model.layers:
            # extract transformer block from layer
            # w_input_layernorm = dequantize_linear(layer.input_layernorm)
            w_input_layernorm = layer.input_layernorm.weight
            w_gate = dequantize_linear(layer.mlp.gate_proj)
            w_up = dequantize_linear(layer.mlp.up_proj)
            w_down = dequantize_linear(layer.mlp.down_proj)
            # w_post_attention_layernorm = dequantize_linear(layer.post_attention_layernorm)
            w_post_attention_layernorm = layer.post_attention_layernorm.weight
            wq = dequantize_linear(layer.self_attn.q_proj)
            wk = dequantize_linear(layer.self_attn.k_proj)
            wv = dequantize_linear(layer.self_attn.v_proj)
            wo = dequantize_linear(layer.self_attn.o_proj)
            bq = layer.self_attn.q_proj.bias
            bk = layer.self_attn.k_proj.bias
            bv = layer.self_attn.v_proj.bias

            layer = Qwen2TransformerBlock(
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=bq,
                bk=bk,
                bv=bv,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=w_input_layernorm,
                w_post_attention_layernorm=w_post_attention_layernorm,
                max_seq_len=max_position_embeddings,
                theta=theta
            )

            self.layers.append(layer)
        
        # norm_weight = dequantize_linear(mlx_model.norm)
        self.norm = RMSNorm(dim=hidden_size, weight=mlx_model.model.norm.weight, eps=rms_norm_eps)
        
        if tie_word_embeddings:
            self.last_linear = lambda x: self.embedding.as_linear(x)
        else:
            lm_head = dequantize_linear(mlx_model.lm_head)
            self.last_linear = lambda x: linear(x=x, w=lm_head)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        x = self.embedding(inputs)
        for layer in self.layers:
            x = layer(x, mask="causal")
        x = self.norm(x)
        return self.last_linear(x)

