import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """
        Compute the RMSNorm given by:

            y = x/sqrt(mean(x^2) + eps) * weight
        
        Args:
            x: [..., self.dim]
        Returns:
            y: [..., self.dim]
        """
        D = x.shape[-1]

        assert D == self.dim

        typ = x.dtype
        x = x.astype(mx.float32)
        x2 = x * x
        x2_mean = mx.mean(x2, axis=-1)
        # Make x2_mean's shape [..., 1] for broadcast
        x2_mean = mx.expand_dims(x2_mean, axis=-1)

        return (x / mx.sqrt(x2_mean + self.eps) * self.weight).astype(typ)
