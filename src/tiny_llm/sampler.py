import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        
        num_logit = logprobs.shape[-1]
        if top_k is not None and top_k < num_logit:
            bottom_k = num_logit - top_k
            indices = mx.argpartition(logprobs, bottom_k, axis=-1)
            # If indices < bottom_k, mask as -mx.inf
            mask = mx.where(indices < bottom_k, -mx.inf, 0)
            logprobs += mask
        
        if top_p is not None and top_p > 0.0 and top_p < 1.0:
            indices = mx.argsort(logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, indices, axis=-1)
            cumsums = mx.cumsum(sorted_logprobs, axis=-1)

            inverse_indices = mx.put_along_axis(
                mx.zeros_like(indices),
                indices,
                mx.arange(indices.shape[-1], dtype=indices.dtype),
                axis=-1,
            )
            cumsums = mx.take_along_axis(cumsums, inverse_indices, axis=-1)

            logprobs = mx.where(cumsums > 1 - top_p, logprobs, -mx.inf)
        
        return mx.random.categorical(logits=logprobs * (1 / temp))

    return sample
