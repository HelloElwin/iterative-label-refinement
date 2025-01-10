import gc

import torch
from transformers import AutoTokenizer


EVALUATOR_PROMPT_TEMPLATE = """QUESTION:
{q}

ANSWER (A):
{a}

ANSWER (B):
{b}

You gave the answer (A) to the question. Do you accept to change it to answer (B) proposed by another model?"""


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    This function returns a tokenizer based on the model name.

    Parameters:
    model_name: The name of the model for which the tokenizer is needed.

    Returns:
    A tokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    return tokenizer


def clear_mem(verbose: bool = False) -> None:
    """
    This function is used to clear the memory allocated by PyTorch.
    It does so by calling the garbage collector to release unused GPU memory.
    After clearing the memory, it prints the current amount of memory still allocated by PyTorch (post-clean).

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    torch.cuda.empty_cache()

    def try_attr(x, a):
        try:
            return getattr(x, a)
        except Exception:
            return None

    if verbose:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)

    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
