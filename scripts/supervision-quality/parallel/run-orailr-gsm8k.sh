#!/bin/bash
. .env  # Load environment variables


###### Setup ######
SEED=0
DATA_SEED=0
DS_NAME=gsm8k
MAX_REPLACE=0.15
WEAK_MODEL_PATH="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=0-lr=64-l=0.0005-ls=cosi_warm-ml=256-ms=gemma-2b-nd=20000-ntd=10000-o=adamw-ri=0-s=$SEED-twd=0-wr=0.05"
EVALUATOR_PATH="$ILR_SAVE_PATH/evaluators/bs=16-dn=d=$DS_NAME-ms=gemma-2b-s=$SEED-e=5-lr=0-l=1e-06-ls=cosi_anne_warm-ms=gemma-2b-o=adamw-sfm=1-s=$SEED-wr=0.05"


###### train round 0 half-data models ######
ROUND=0
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
python -m iterative_label_refinement.experiments.ilr \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=weak-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=weak-wms=gemma-2b" \
 --evaluator_path=$EVALUATOR_PATH \
 --max_replace=$MAX_REPLACE \
 --use_oracle_feedback=True \
 --round_id=$ROUND \
 --seed=$SEED


###### train round 1 half-data models ######
ROUND=1
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
python -m iterative_label_refinement.experiments.ilr \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --evaluator_path=$EVALUATOR_PATH \
 --max_replace=$MAX_REPLACE \
 --use_oracle_feedback=True \
 --round_id=$ROUND \
 --seed=$SEED


###### train round 1 and round 2 full-data models ######
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=1 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=2 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait


###### train round 2 half-data models ######
ROUND=2
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
python -m iterative_label_refinement.experiments.ilr \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --evaluator_path=$EVALUATOR_PATH \
 --max_replace=$MAX_REPLACE \
 --use_oracle_feedback=True \
 --round_id=$ROUND \
 --seed=$SEED


###### train round 3 half-data models ######
ROUND=3
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
python -m iterative_label_refinement.experiments.ilr \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --evaluator_path=$EVALUATOR_PATH \
 --max_replace=$MAX_REPLACE \
 --use_oracle_feedback=True \
 --round_id=$ROUND \
 --seed=$SEED


###### train round 3 and round 4 full-data models ######
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=3 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=4 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait


###### train round 4 half-data models ######
ROUND=4
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
python -m iterative_label_refinement.experiments.ilr \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --evaluator_path=$EVALUATOR_PATH \
 --max_replace=$MAX_REPLACE \
 --use_oracle_feedback=True \
 --round_id=$ROUND \
 --seed=$SEED


###### train round 5 half-data models ######
ROUND=5
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
python -m iterative_label_refinement.experiments.ilr \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=2-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=orailr${MAX_REPLACE}-wms=gemma-2b" \
 --evaluator_path=$EVALUATOR_PATH \
 --max_replace=$MAX_REPLACE \
 --use_oracle_feedback=True \
 --round_id=$ROUND \
 --seed=$SEED


###### train round 5 and round 6 full-data models ######
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=5 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
sleep 3m
python -m iterative_label_refinement.experiments.main --method="orailr${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=2 --round_id=6 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH &
wait
