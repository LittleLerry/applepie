from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import json
import re

def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = logits - torch.logsumexp(logits, dim=-1,keepdim=True)
    entropy = - torch.exp(log_probs) * log_probs
    return entropy.sum(dim=-1)

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    assert(len(prompt_strs) == len(output_strs))
    batch_size = len(prompt_strs)

    if tokenizer.eos_token_id is None:
        raise ValueError("No EOS token for this tokenizer.")
    
    prompt_tokenized = tokenizer(prompt_strs, add_special_tokens=False, return_tensors=None)["input_ids"]
    output_tokenized = tokenizer(output_strs, add_special_tokens=False, return_tensors=None)["input_ids"]

    temp = []
    for t1,t2 in zip(prompt_tokenized , output_tokenized):
        temp.append(t1 + t2)
    seq_len = max([len(t) for t in temp])

    result = [t + [tokenizer.eos_token_id] * (seq_len - len(t)) for t in temp]
    result = torch.tensor(result)

    input_ids = result[:,:-1]
    labels = result[:,1:]

    starts = torch.tensor([len(prompt)-1 for prompt in prompt_tokenized]).unsqueeze(1).expand(-1, seq_len - 1)
    ends = torch.tensor([len(t)-1 for t in temp]).unsqueeze(1).expand(-1, seq_len - 1)
    response_mask = torch.arange(seq_len - 1).unsqueeze(0).expand(batch_size, -1)
    response_mask = ((response_mask >= starts) & (response_mask < ends)) * 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    l = model(input_ids).logits
    log_probs = l - torch.logsumexp(l, dim=-1,keepdim=True)

    b, s = labels.shape

    result = log_probs[torch.arange(b)[:, None], torch.arange(s), labels]

    result_dict = {}
    result_dict["log_probs"] = result
    if (return_token_entropy):
        result_dict["token_entropy"] = run_compute_entropy(l)

    return result_dict