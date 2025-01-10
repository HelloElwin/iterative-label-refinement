#!/bin/bash
. .env  # Load environment variables

SEED=0
EPOCH=4
DATA_SEED=0
DS_NAME=saferpaca

python -m iterative_label_refinement.experiments.main --saferpaca_mode=1 --save_frequency=0.1 --model_size=google/gemma-2b --ds_name=$DS_NAME --epochs=$EPOCH --generate_weak_labels=False --do_eval=False --seed=$SEED --data_seed=$DATA_SEED
python -m iterative_label_refinement.experiments.main --saferpaca_mode=2 --save_frequency=0.1 --model_size=google/gemma-2b --ds_name=$DS_NAME --epochs=$EPOCH --generate_weak_labels=False --do_eval=False --seed=$SEED --data_seed=$DATA_SEED
python -m iterative_label_refinement.experiments.generate_evaluator_data --model_size=google/gemma-2b --ds_name=$DS_NAME --seed=$SEED -do_eval=False --save_path="$ILR_SAVE_PATH/default/bs=32-ds=0-dn=$DS_NAME-e=$EPOCH-hd=0-lr=64-l=0.0005-ls=cosi_warm-ml=512-ms=gemma-2b-nd=20000-ntd=10000-o=adamw-ri=0-sm=1-sf=0.1-s=0-twd=0-wr=0.05"
python -m iterative_label_refinement.core.train_evaluator --model_size=google/gemma-2b --seed=$SEED --ds_name="$ILR_SAVE_PATH/evaluator-data/d=$DS_NAME-ms=gemma-2b-s=0"