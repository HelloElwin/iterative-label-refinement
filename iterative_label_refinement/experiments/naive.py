import os
import random

import datasets
import fire
import numpy as np

from iterative_label_refinement.core.logger import logger


def main(
    ds_name: str,
    weak_model_path: str,
    model1_path: str,
    model2_path: str,
    round_id: int,
    save_results: bool = True,
    max_replace: float = 0.1,
    seed: int = 0,
    start_with_correct_labels: bool = False,
) -> None:
    if not (f"ri={round_id}" in model1_path and f"ri={round_id}" in model2_path):
        raise ValueError("Round mismatch.")

    # print setup
    logger.info("Starting to refine weak labels")
    logger.info(f" - weak model path: {weak_model_path}")
    logger.info(f" - model1 path: {model1_path}")
    logger.info(f" - model2 path: {model2_path}")
    logger.info(f" - dataset: {ds_name}")
    logger.info(f" - round: {round_id}")
    logger.info(f" - save: {save_results}")
    logger.info(f" - maximal replacement rate: {max_replace}")
    logger.info(f" - start with correct labels: {start_with_correct_labels}")
    logger.info(f" - seed: {seed}")

    # setup random seeds
    random.seed(seed)
    np.random.seed(seed)

    method = "cor" * start_with_correct_labels + "naive" + str(max_replace)

    # load labels from the previous round and new proposals
    if round_id > 0:
        ds_0 = datasets.load_from_disk(os.path.join(weak_model_path, f"{method}_{round_id - 1}_weak_labels"))
    else:
        ds_0 = datasets.load_from_disk(os.path.join(weak_model_path, "weak_labels"))
        if start_with_correct_labels:
            ds_0 = ds_0.filter(lambda x: x["acc"] > 0)
    ds_1 = datasets.load_from_disk(os.path.join(model1_path, "weak_labels"))
    ds_2 = datasets.load_from_disk(os.path.join(model2_path, "weak_labels"))

    ds_proposals = datasets.concatenate_datasets([ds_1, ds_2])

    ds_0 = sorted(ds_0, key=lambda x: x["question"])
    ds_proposals = sorted(ds_proposals, key=lambda x: x["question"])

    replace_idx = np.random.choice(len(ds_0), int(len(ds_0) * max_replace), replace=False)
    ds_refined = [ds_proposals[i] if i in replace_idx else ds_0[i] for i in range(len(ds_0))]
    ds_refined = datasets.Dataset.from_list(ds_refined)
    ds_refined = ds_refined.shuffle()
    logger.info("Done naive refinement.")

    if save_results:
        ds_refined.save_to_disk(os.path.join(weak_model_path, f"{method}_{round_id}_weak_labels"))
        logger.info(f"Saved refined data to {weak_model_path}/{method}_{round_id}_weak_labels")


if __name__ == "__main__":
    fire.Fire(main)
