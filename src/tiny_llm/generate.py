import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        """
        Takes a list of tokens, return the logits for all the input
        tokens.

        Args:
            model: Qwen2ModelWeek1
            y: [N.., S] (...batch size, sequence length)
        Returns:
            logits: [N..., S, vocab_size]
        """

        output_logits = model(y)
        # [N..., 1, vocab_size]
        if len(output_logits.shape) == 2:
            last_logits = output_logits[-1, :]
        else:
            last_logits = output_logits[:, -1, :]
        res = sampler(last_logits)
        return res.item()


    
    tokens = tokenizer.encode(prompt)
    next_token = _step(model, tokens)
    tokenizer._detokenizer.reset()
    while next_token not in tokenizer._eos_token_ids:
        tokens.append(next_token)
        tokenizer._detokenizer.add_token(next_token)
        print(tokenizer._detokenizer.last_segment, end="", flush=True)
        next_token = _step(model, tokens)
    
    tokenizer._detokenizer.finalize()
    return tokenizer._detokenizer.text
    




def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
