import os
import openai
import random
from typing import List, Optional

import fire
import torch
import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from iterative_label_refinement.core.logger import logger
from iterative_label_refinement.core.common import EVALUATOR_PROMPT_TEMPLATE
from iterative_label_refinement.core.datasets import (
    DATASET_EVAL_FUNC,
    load_dataset,
)


def run_inference(evaluator_path: str, prompts: List[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b" if "2b" in evaluator_path else "mistralai/Mistral-7B-v0.1")
    config = AutoConfig.from_pretrained("google/gemma-2b" if "gemma-2b" in evaluator_path else "mistralai/Mistral-7B-v0.1")
    model = AutoModelForSequenceClassification.from_pretrained(
        evaluator_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to("cuda")

    model.eval()
    all_outputs = []
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Running inference"):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            all_outputs.append(output.logits.cpu().float())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_probs = torch.nn.functional.softmax(torch.tensor(all_outputs), dim=1).numpy()

    return all_probs


def get_embedding(client: openai.OpenAI, text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    text = text.replace("\n", " ")
    for _ in range(3):
        try:
            embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
            break
        except Exception:
            continue
    return embedding


def main(
    ds_name: str,
    weak_model_path: str,
    model1_path: str,
    model2_path: str,
    evaluator_path: str,
    round_id: int,
    save_results: bool = True,
    max_replace: float = 0.1,
    seed: int = 0,
    use_oracle_feedback: bool = False,
    start_with_correct_labels: bool = False,
) -> None:
    if not (f"ri={round_id}" in model1_path and f"ri={round_id}" in model2_path):
        raise ValueError("Round mismatch.")

    eval_func = DATASET_EVAL_FUNC[ds_name]
    _ = load_dataset(ds_name, seed, split_sizes={"train": 20000, "test": 10000})  # for initialization

    # print setup
    logger.info("Starting to refine weak labels")
    logger.info(f" - evaluator path: {evaluator_path}")
    logger.info(f" - weak model path: {weak_model_path}")
    logger.info(f" - model1 path: {model1_path}")
    logger.info(f" - model2 path: {model2_path}")
    logger.info(f" - dataset: {ds_name}")
    logger.info(f" - round: {round_id}")
    logger.info(f" - save: {save_results}")
    logger.info(f" - maximal replacement rate: {max_replace}")
    logger.info(f" - use oracle feedback: {use_oracle_feedback}")
    logger.info(f" - start with correct labels: {start_with_correct_labels}")
    logger.info(f" - seed: {seed}")

    # setup random seeds
    random.seed(seed)
    np.random.seed(seed)

    method = "cor" * start_with_correct_labels + "ora" * use_oracle_feedback + "ilr" + str(max_replace)

    # load labels from the previous round and new proposals
    if round_id > 0:
        ds_0 = datasets.load_from_disk(os.path.join(weak_model_path, f"{method}_{round_id - 1}_weak_labels"))
    else:
        ds_0 = datasets.load_from_disk(os.path.join(weak_model_path, "weak_labels"))
        if start_with_correct_labels:
            ds_0 = ds_0.filter(lambda x: x["acc"] > 0)
    ds_1 = datasets.load_from_disk(os.path.join(model1_path, "weak_labels"))
    ds_2 = datasets.load_from_disk(os.path.join(model2_path, "weak_labels"))
    weak_answers = {x["question"]: x["answer"] for x in ds_0}
    weak_accs = {x["question"]: x["acc"] for x in ds_0}

    ds_proposals = datasets.concatenate_datasets([ds_1, ds_2])
    ds_proposals = ds_proposals.filter(lambda x: x["question"] in weak_answers)
    ds_refined = []

    all_questions = [x["question"] for x in ds_proposals]
    all_gt = [x["gt_answer"] for x in ds_proposals]
    all_strong = [x["answer"] for x in ds_proposals]
    all_weak = [weak_answers[x["question"]] for x in ds_proposals]
    all_weak_acc = [weak_accs[x["question"]] for x in ds_proposals]

    if "paca" in ds_name:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("Please set OPENAI_API_KEY in the environment variable.")
        client = openai.OpenAI(api_key=api_key)
        embedded_pairs = []
        for x, y in tqdm(zip(all_weak, all_strong), desc="Computing embeddings"):
            try:
                embedded_pairs.append((get_embedding(client, x), get_embedding(client, y)))
            except Exception:
                embedded_pairs.append((np.zeros(512), np.zeros(512)))
        similarities = [np.dot(x, y) for x, y in embedded_pairs]  # OpenAI API returns normalized embeddings
        agree = [x > np.percentile(similarities, 50) for x in similarities]
    elif ds_name == "bird":  # BIRD uses batch evaluation
        agree = eval_func(all_questions, all_strong, all_weak)
        agree = [x == 1 for x in agree]
    else:
        agree = [eval_func(q, a_s, a_w) for q, a_s, a_w in zip(all_questions, all_strong, all_weak)]
        agree = [x == 1 for x in agree]

    eval_set = []
    prompts = []
    prompts_inv = []
    for i in range(len(ds_proposals)):
        if agree[i]:
            ds_refined.append(
                {
                    "question": all_questions[i],
                    "answer": all_weak[i],
                    "gt_answer": all_gt[i],
                    "score": -1.0,
                    "acc": float(all_weak_acc[i]),
                }
            )
        else:
            eval_set.append(ds_proposals[i])
            prompts.append(EVALUATOR_PROMPT_TEMPLATE.format(q=all_questions[i], a=all_weak[i], b=all_strong[i]))
            prompts_inv.append(EVALUATOR_PROMPT_TEMPLATE.format(q=all_questions[i], a=all_strong[i], b=all_weak[i]))

    shuffle_idx = np.random.permutation(len(eval_set))
    eval_set = [eval_set[i] for i in shuffle_idx]
    prompts = [prompts[i] for i in shuffle_idx]
    prompts_inv = [prompts_inv[i] for i in shuffle_idx]
    all_prompts = prompts + prompts_inv

    if use_oracle_feedback:
        scores = [x["acc"] for x in eval_set]
        threshold = 0.5
    else:
        all_probs = run_inference(evaluator_path, all_prompts)
        probs, probs_inv = all_probs[: len(prompts)], all_probs[len(prompts) :]
        scores = [np.mean([x[1], y[0]]) for x, y in zip(probs, probs_inv)]
        threshold = max(0.5, np.percentile(scores, 100 * (1 - max_replace)))

    for x, score in zip(eval_set, scores):
        q = x["question"]
        a_strong = x["answer"]
        a_weak = weak_answers[q]
        a_gt = x["gt_answer"]
        acc_weak = weak_accs[q]
        acc_strong = x["acc"]

        accept = bool(score > threshold)
        ds_refined.append(
            {
                "question": q,
                "answer": a_strong if accept else a_weak,
                "gt_answer": a_gt,
                "score": float(score),
                "acc": float(acc_strong if accept else acc_weak),
            }
        )

    ds_refined = datasets.Dataset.from_list(ds_refined)
    ds_refined = ds_refined.shuffle(seed=seed)
    logger.info("Done refinement.")

    if save_results:
        ds_refined.save_to_disk(os.path.join(weak_model_path, f"{method}_{round_id}_weak_labels"))
        logger.info(f"Saved refined data to {weak_model_path}/{method}_{round_id}_weak_labels")


if __name__ == "__main__":
    fire.Fire(main)
