import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # Assume dimension is even in the course
        assert dims % 2 == 0

        w = 1 / (base ** (mx.arange(dims // 2) * 2 / dims))

        self.dims = dims
        self.traditional = traditional
        self.seq_len = seq_len

        seq = mx.arange(seq_len+1)
        freq = mx.outer(seq, w)
        # sin_freq: [seq_len, d // 2]
        self.sin_freq = mx.sin(freq)
        self.cos_freq = mx.cos(freq)

    def __call__(
        self,
        x: mx.array, # [N, L, H, D]
        offset: list[slice] | slice | None = None
    ) -> mx.array:
        if self.traditional:
            return self.encode_traditional(x, offset=offset)
        return self.encode_non_traditional(x, offset=offset)
    
    def encode_traditional(
        self,
        x: mx.array,
        offset: list[slice] | slice | None = None
    ) -> mx.array:
        L = x.shape[-3]
        H = x.shape[-2]
        D = x.shape[-1]
        batches = x.shape[:-3]

        assert D == self.dims
        assert L <= self.seq_len

        sin, cos = self._extact_sin_cos(L, offset=offset)
        
        x = mx.reshape(x, list(batches) + [L, H, D // 2, 2])

        # Even indices
        x0 = x[..., 0]
        # Odd indices
        x1 = x[..., 1]

        # broadcast for each batch and each head
        sin = mx.reshape(sin, [1, L, 1, D // 2])
        cos = mx.reshape(cos, [1, L, 1, D // 2])

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        out = mx.stack([out0, out1], axis=-1)
        out = mx.reshape(out, list(batches) + [L, H, D])
        return out
    
    def encode_non_traditional(
        self,
        x: mx.array,
        offset: list[slice] | slice | None = None
    ) -> mx.array:
        L = x.shape[-3]
        H = x.shape[-2]
        D = x.shape[-1]
        batches = x.shape[:-3]

        assert D == self.dims
        assert L <= self.seq_len

        sin, cos = self._extact_sin_cos(L, offset=offset)
       
        # Instead of paring between i and i+1,
        # non-traditional RoPE can be viewed as paring between
        # i and D//2 + i + 1
        x = mx.reshape(x, list(batches) + [L, H, 2, D//2])

        # First Half
        x0 = x[...,0,:]
        # Second Half
        x1 = x[...,1,:]

        sin = mx.reshape(sin, [1, L, 1, D//2])
        cos = mx.reshape(cos, [1, L, 1, D//2])

        out1 = x0 * cos - x1 * sin
        out2 = x0 * sin + x1 * cos

        out = mx.concatenate([out1, out2], -1)
        out = mx.reshape(out, list(batches) + [L, H, D])
        return out
    
    def _extact_sin_cos(self, L: int, offset: list[slice] | slice | None = None,) -> tuple[mx.array, mx.array]:
        if offset is None:
            sin = self.sin_freq[:L]
            cos = self.cos_freq[:L]
        else:
            if isinstance(offset, slice):
                sin = self.sin_freq[offset]
                cos = self.cos_freq[offset]
            else:
                raise Exception("not implemented")
        
        return sin, cos
        
        
