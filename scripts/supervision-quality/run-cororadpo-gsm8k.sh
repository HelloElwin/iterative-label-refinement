#!/bin/bash
. .env  # Load environment variables


###### Setup ######
SEED=0
DATA_SEED=0
DS_NAME=gsm8k
BETA=0.1
SUBSAMPLE=0.15
WEAK_MODEL_PATH="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=0-lr=64-l=0.0005-ls=cosi_warm-ml=256-ms=gemma-2b-nd=20000-ntd=10000-o=adamw-ri=0-s=$SEED-twd=0-wr=0.05"
EVALUATOR_PATH="$ILR_SAVE_PATH/evaluators/bs=16-dn=d=$DS_NAME-ms=gemma-2b-s=$SEED-e=5-lr=0-l=1e-06-ls=cosi_anne_warm-ms=gemma-2b-o=adamw-sfm=1-s=$SEED-wr=0.05"


ROUND=0
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH 
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=weak-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED


ROUND=1
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH \
    --dpo_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$((ROUND-1))-s=$SEED-twd=0-wr=0.05-wlt=weak-wms=gemma-2b"
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED


ROUND=2
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH \
    --dpo_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$((ROUND-1))-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b/train_adapter"
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED


ROUND=3
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH \
    --dpo_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$((ROUND-1))-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b/train_adapter"
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED


ROUND=4
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH \
    --dpo_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$((ROUND-1))-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b/train_adapter"
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED


ROUND=5
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH \
    --dpo_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$((ROUND-1))-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b/train_adapter"
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED


ROUND=6
python -m iterative_label_refinement.experiments.main --method="cororadpo${BETA}${SUBSAMPLE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --label_temp=0.7 --round_id=$ROUND --seed=$SEED --data_seed=$DATA_SEED --weak_labels_path=$WEAK_MODEL_PATH \
    --dpo_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$((ROUND-1))-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b/train_adapter"
python -m iterative_label_refinement.experiments.dpo \
    --ds_name=$DS_NAME \
    --weak_model_path=$WEAK_MODEL_PATH \
    --strong_model_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dd=1-db=$BETA-dlt=sigmoid-dn=$DS_NAME-e=2-hd=0-lr=64-l=1e-06-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=rmsprop-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=cororadpo${BETA}${SUBSAMPLE}-wms=gemma-2b" \
    --evaluator_path=$EVALUATOR_PATH \
    --use_oracle_feedback=True \
    --start_with_correct_labels=True \
    --subsample=$SUBSAMPLE \
    --round_id=$ROUND \
    --seed=$SEED
