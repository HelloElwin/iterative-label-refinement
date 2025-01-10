import time
from typing import Callable, Optional

import datasets
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from iterative_label_refinement.core.logger import logger


def eval_model_vllm(
    model: LLM,
    ds: datasets.Dataset,
    max_len: int,
    eval_func: Callable,
    eval_batch_size: int = 1,
    temperature: float = 0.0,
    num_beams: int = 4,
    num_samples: int = 1,
    lora_path: Optional[str] = None,
    top_k: int = -1,
    top_p: float = 1.0,
) -> datasets.Dataset:
    """
    This function evaluates a model on a dataset using vLLM.
    It is also used to sample responses for DPO and proposals for label refinement in ILR.
    """
    if eval_batch_size != 1:
        raise NotImplementedError("Batch evaluation not supported yet")

    if num_samples > 1 and temperature == 0:
        logger.warning("Temperature is 0, but num_samples > 1. Setting temperature to 1.0")
        temperature = 1.0

    prompts = ["USER:\n" + q + "\n\nASSISTANT:\n" for q in ds["question"]]
    sampling_params = SamplingParams(
        n=num_samples,
        best_of=num_beams if temperature == 0 else num_samples,
        use_beam_search=(temperature == 0),
        top_k=top_k if temperature else -1,
        top_p=top_p if temperature else 1,
        max_tokens=max_len,
        temperature=temperature,
        skip_special_tokens=True,
        seed=0,
    )
    t0 = time.time()
    outputs = model.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path) if lora_path else None,
    )
    logger.info(f"Generation took: {time.time() - t0} seconds")

    results = []
    questions = []
    pred_answers = []
    pred_tokens = []
    gt_answers = []
    for x, output in zip(ds, outputs):
        # store flattened results because BIRD needs batched evaluation (not elegant but anyway)
        generated_texts = [out.text.strip() for out in output.outputs]
        generated_tokens = [out.token_ids for out in output.outputs]
        questions.extend([x["question"]] * num_samples)
        pred_answers.extend(generated_texts)
        pred_tokens.extend(generated_tokens)
        gt_answers.extend([x.get("gt_answer", x["answer"])] * num_samples)

    if eval_func.__doc__ and "batch" in eval_func.__doc__:
        accs = eval_func(questions, pred_answers, gt_answers)
    else:
        accs = [eval_func(q, pred, gt) for q, gt, pred in zip(questions, gt_answers, pred_answers)]

    if num_samples > 1:
        # gather results back into lists
        accs = [accs[i * num_samples : (i + 1) * num_samples] for i in range(len(ds))]
        pred_answers = [pred_answers[i * num_samples : (i + 1) * num_samples] for i in range(len(ds))]
        pred_tokens = [pred_tokens[i * num_samples : (i + 1) * num_samples] for i in range(len(ds))]

    results = [
        dict(
            question=ds[i]["question"],
            gt_answer=ds[i].get("gt_answer", ds[i]["answer"]),
            answer=pred_answers[i],
            answer_tokens=pred_tokens[i],
            acc=accs[i],
        )
        for i in range(len(ds))
    ]

    # compute average accuracy
    accs = [np.mean(r["acc"]) for r in results]
    logger.info(f"Accuracy: {np.mean(accs)} +/- {np.std(accs) / np.sqrt(len(accs))}")

    return datasets.Dataset.from_list(results)
