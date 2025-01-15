# Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision

This repository contains code and data for the paper [Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision](https://arxiv.org/abs/2501.07886).


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/helloelwin/iterative-label-refinement.git
    cd iterative-label-refinement
    ```

2. Create virtual environment and install dependencies (recommended Python version=3.11):
    ```bash
    conda create -n ilr python=3.11
    conda activate ilr
    pip install -r requiremntes.txt
    ```

3. Install the `iterative_label_refinement` package:
    ```bash
    pip install -e .
    ```

4. Set up environment variables:
    ```bash
    cp .env.example .env
    vim .env  # edit this file with your own values
    ```


## Models and datasets

We use HuggingFace's `transformers` to load pre-trained language models, so please make sure you have permission to access them and you have logged in locally. Models are registered in `iterative_label_refinement/core/models.py`. 

Datasets are stored in `data/` and registered in `iterative_label_refinement/core/datasets.py`. To run experiments on the BIRD dataset, you need to additionally download the databases for code execution from [here](https://bird-bench.github.io) and specify where they are stored in your `.env` file.


## Experiments

### Training the LM evaluator

Before running ILR or DPO, we need to train the LM evaluator that simulates unreliable comparison feedback. The complete scripts are in `scripts/evaluator-training`. Take GSM8K as an example, simply run
```bash
./scripts/evaluator-training/train-evaluator-gsm8k.sh
```

The following is a breakdown of how the script trains the evaluator:

1. Training and saving checkpoints of a small LM finetuned on GT.

    ```bash
    python -m iterative_label_refinement.experiments.main \
        --ds_name=gsm8k \
        --save_frequency=0.1 \  # save checkpoints every 10% of the training steps
        --model_size=google/gemma-2b
    ```

2. Inferencing the checkpoints on the training data and gathering the results to construct a pairwise comparison dataset.

    ```bash
    python -m iterative_label_refinement.experiments.generate_evaluator_data \
        --ds_name=gsm8k \
        --model_size=google/gemma-2b \
        --save_path=PATH_TO_THE_MODEL_CHECKPOINTS  # this can be found in the output of step 1
    ```

2. Train the evaluator on the pairwise comparison dataset.

    ```bash
    python -m iterative_label_refinement.core.train_evaluator \
        --model_size=google/gemma-2b \
        --ds_name=PATH_TO_THE_GENERATED_DATASET  # this can be found in the output step 2
    ```


### Running SFT+ILR

We present complete scripts for running multiple rounds of ILR in `scripts/main-experiments`. Take GSM8K training as an example, simply run
```bash
./scripts/main-experiments/run-ilr-gsm8k.sh
```

(If you are training 7B models with A100 or other GPUs with larger memory, we suggest using scripts in `scripts/main-experiments/parallel` to run two jobs parallelly for each round)

The following is a breakdown of how the script runs SFT+ILR in the first round:

1. Train the small LM to simulate unreliable demonstrations.
    ```bash
    python -m iterative_label_refinement.experiments.main \
        --ds_name=gsm8k \
        --model_size=google/gemma-2b \
        ...
    ```

2. Train the initial SFT model.
    ```bash
    python -m iterative_label_refinement.experiments.main \
        --ds_name=gsm8k \
        --model_size=mistralai/Mistral-7B-v0.1 \
        --weak_labels_path=PATH_TO_UNRELIABLE_DEMONSTRATIONS \  # this can be found in the output of step 1
        ...
    ```

2. Train half-data models and get proposals
    ```bash
    python -m iterative_label_refinement.experiments.main \
        --ds_name=gsm8k \
        --model_size=mistralai/Mistral-7B-v0.1 \
        --method=ilr0.15 \  # 0.15 represents the maximum number of labels to updated
        --round_id=1 \
        --half_data=1 \  # train on the first half
        --weak_labels_path=PATH_TO_UNRELIABLE_DEMONSTRATIONS \
        ...

    python -m iterative_label_refinement.experiments.main \
        --ds_name=gsm8k \
        --model_size=mistralai/Mistral-7B-v0.1 \
        --method=ilr0.15 \ 
        --round_id=1 \
        --half_data=2 \  # train on the second half
        --weak_labels_path=PATH_TO_UNRELIABLE_DEMONSTRATIONS \
        ...
    ```

3. Perform label refinement using unreliable comparison feedback.
    ```bash
    python -m iterative_label_refinement.experiments.ilr \
        --ds_name=gsm8k \
        --round_id=1 \
        --max_replace=0.15 \
        --weak_model_path=PATH_TO_THE_UNRELIABLE_SUPERVISOR \
        --model1_path=PATH_TO_THE_FIRST_HALF_DATA_MODEL \
        --model2_path=PATH_TO_THE_SECOND_HALF_DATA_MODEL \
        --evaluator_path=PATH_TO_THE_EVALUATOR \
        ...
    ```

4. Run SFT on feedback data.
    ```bash
    python -m iterative_label_refinement.experiments.main 
        --ds_name=gsm8k \
        --model_size=mistralai/Mistral-7B-v0.1 \
        --method=ilr0.15 \
        --round_id=1 \
        --weak_labels_path=PATH_TO_REFINED_DATA \  # this can be found in the output of step 3
        ...
    ```

The rest rounds follow the same procedure and repeat steps 2-4.


### Running SFT+DPO

We present complete scripts for running multiple rounds of DPO in `scripts/main-experiments`. Take GSM8K as an example, simply run
```bash
./scripts/main-experiments/run-dpo-gsm8k.sh
```

The following is a breakdown of how the script runs SFT+DPO in the first round:

1. Train the small LM to simulate unreliable demonstrations. (skip if done in the ILR experiment)
    ```bash
    python -m iterative_label_refinement.experiments.main \
        --ds_name=gsm8k \
        --model_size=google/gemma-2b \
        ...
    ```

2. Train the initial SFT model and sample pairs of responses.
    ```bash
    python -m iterative_label_refinement.experiments.main
        --ds_name=gsm8k \
        --method=dpo0.10.15 \  # 0.1 represents beta, 0.15 represents the subsample rate
        --model_size=mistralai/Mistral-7B-v0.1 \
        --label_temp=0.7 \  # temperature for sampling responses
        --weak_labels_path=PATH_TO_UNRELIABLE_DEMONSTRATIONS \
        ...
    ```

3. Label preference using unreliable comparison feedback.
    ```bash
    python -m iterative_label_refinement.experiments.dpo \
        --ds_name=gsm8k \
        --round_id=1 \
        --subsample=0.15 \
        --weak_model_path=PATH_TO_THE_UNRELIABLE_SUPERVISOR \
        --strong_model_path=PATH_TO_THE_LAST_MODEL \  # this can be found in the output of step 2
        --evaluator_path=PATH_TO_THE_EVALUATOR \
        ...
    ```

4. Run DPO on new data
    ```bash
    python -m iterative_label_refinement.experiments.main 
        --ds_name=gsm8k \
        --model_size=mistralai/Mistral-7B-v0.1 \
        --method=dpo0.10.15 \
        --round_id=1 \
        --label_temp=0.7 \
        --weak_labels_path=PATH_TO_DPO_DATA \  # this can be found in the output of step 3
	    --dpo_path=PATH_TO_THE_LAST_MODEL \
        ...
    ```

The rest rounds follow the same procedure and repeat steps 3-4.

### Running naive ILR

To run naive ILR (i.e., replacing labels without using comparison feedback), you can use the use a similar pipeline as SFT+ILR, but replace the `ilr` in commands and arguments with `naive`. For example, the following script runs naive ILR on GSM8K:
```bash
./scripts/main-experiments/run-naive-gsm8k.sh
```

### Experiments with higher supervision quality

We provide scripts for running SFT+ILR and SFT+DPO in two other settings in `scripts/supervision-quality`:

1. Unreliable demonstrations + reliable comparison feedback: use scipts with name `run-ora*-*.sh`.
2. Reliable demonstrations + reliable comparison feedback:  use scipts with name `run-corora*-*.sh`.

## Citation

```bibtex
@misc{ye2025iterative,
  title={Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision}, 
  author={Yaowen Ye and Cassidy Laidlaw and Jacob Steinhardt},
  year={2025},
  eprint={2501.07886},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2501.07886}, 
}
```


## Acknowledgments

The codebase is based on [openai/weak-to-strong](https://github.com/openai/weak-to-strong).
