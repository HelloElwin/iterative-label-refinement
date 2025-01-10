#!/usr/bin/zsh
. .env  # Load environment variables


###### Setup ######
SEED=0
EPOCH=2
DATA_SEED=0
DS_NAME=gsm8k
MAX_REPLACE=0.15
WEAK_MODEL_PATH="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=0-lr=64-l=0.0005-ls=cosi_warm-ml=256-ms=gemma-2b-nd=20000-ntd=10000-o=adamw-ri=0-s=$SEED-twd=0-wr=0.05"


###### train the small unreliable model ######
# python -m iterative_label_refinement.experiments.main --model_size=google/gemma-2b --ds_name=$DS_NAME --epochs=$EPOCH --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35


###### train stage 0 half-data models ######
ROUND=0
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.naive \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=weak-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=weak-wms=gemma-2b" \
 --max_replace=$MAX_REPLACE \
 --round_id=$ROUND \
 --seed=$SEED


###### train stage 1 half-data models ######
ROUND=1
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.naive \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --max_replace=$MAX_REPLACE \
 --round_id=$ROUND \
 --seed=$SEED


###### train stage 1 and stage 2 full-data models ######
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=1 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=2 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH


###### train stage 2 half-data models ######
ROUND=2
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.naive \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --max_replace=$MAX_REPLACE \
 --round_id=$ROUND \
 --seed=$SEED


###### train stage 3 half-data models ######
ROUND=3
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.naive \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --max_replace=$MAX_REPLACE \
 --round_id=$ROUND \
 --seed=$SEED


###### train stage 3 and stage 4 full-data models ######
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=3 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=4 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH


###### train stage 4 half-data models ######
ROUND=4
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.naive \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --max_replace=$MAX_REPLACE \
 --round_id=$ROUND \
 --seed=$SEED


###### train stage 5 half-data models ######
ROUND=5
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=1 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=$ROUND --half_data=2 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.naive \
 --ds_name=$DS_NAME \
 --weak_model_path=$WEAK_MODEL_PATH \
 --model1_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=1-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --model2_path="$ILR_SAVE_PATH/default/bs=32-ds=$DATA_SEED-dn=$DS_NAME-e=$EPOCH-hd=2-lr=64-l=0.0001-ls=cosi_warm-ml=256-ms=Mistral-7B-v0.1-nd=20000-ntd=10000-o=adamw-ri=$ROUND-s=$SEED-twd=0-wr=0.05-wlt=naive${MAX_REPLACE}-wms=gemma-2b" \
 --max_replace=$MAX_REPLACE \
 --round_id=$ROUND \
 --seed=$SEED


###### train stage 5 and stage 6 full-data models ######
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=5 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
python -m iterative_label_refinement.experiments.main --method="naive${MAX_REPLACE}" --model_size=mistralai/Mistral-7B-v0.1 --ds_name=$DS_NAME --epochs=$EPOCH --round_id=6 --half_data=0 --seed=$SEED --data_seed=$DATA_SEED --gpu_usage=0.35 --weak_labels_path=$WEAK_MODEL_PATH
