#!/bin/bash
echo "version: 1.0.2"

PROJECT_DIR=$(pwd)
CACHE_DIR=$PROJECT_DIR/saved_models/base_model/models--microsoft--codebert-base

# Set default values
DATA_PATH=${DATA_PATH:-"$PROJECT_DIR/data"}
OUTPUT_PATH=${OUTPUT_PATH:-"$PROJECT_DIR/output"}

echo $DATA_PATH
echo $OUTPUT_PATH


TEST_NAME=${TEST_NAME:-"test.jsonl"}
TRAIN_NAME=${TRAIN_NAME:-"train.jsonl"}
VALID_NAME=${VALID_NAME:-"valid.jsonl"}

MODEL_NAME=${MODEL_NAME:-ContraBERT_G}
PRETRAIN_DIR=${PRETRAIN_DIR:-"$PROJECT_DIR/saved_models/pretrain_models"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-""}

EPOCH=${EPOCH:-10}
BLOCK_SIZE=${BLOCK_SIZE:-400}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
SEED=${SEED:-123456}

# Create a directory for extraction
EXTRACT_DIR="$PROJECT_DIR/data"


# Check if DATA_PATH points to a .gz archive
if [[ $DATA_PATH == *.gz ]]; then
  # If it does, extract the archive to the project's data directory
  mkdir -p $EXTRACT_DIR
  tar -xzf $DATA_PATH -C $EXTRACT_DIR --strip-components=0
  DATA_PATH=$EXTRACT_DIR
  echo "Unziped archive:"
  ls $DATA_PATH
fi

# Create output directory
mkdir -p ${OUTPUT_PATH}

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
# Create a new directory with the current time
mkdir "$OUTPUT_PATH/experiment_${EXPERIMENT_NAME}_$current_time"

cd defect_detection
# Run the model
python3 vulnerability_detection.py \
  --output_dir=${OUTPUT_PATH}/"experiment_${EXPERIMENT_NAME}_$current_time" \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=${PRETRAIN_DIR}/$MODEL_NAME \
  --do_eval \
  --do_train \
  --do_test \
  --train_data_file=$DATA_PATH/$TRAIN_NAME \
  --eval_data_file=$DATA_PATH/$VALID_NAME \
  --test_data_file=$DATA_PATH/$TEST_NAME \
  --epoch $EPOCH \
  --block_size $BLOCK_SIZE \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --max_grad_norm $MAX_GRAD_NORM \
  --evaluate_during_training \
  --cache_dir=$CACHE_DIR \
  --seed $SEED 2>&1| tee ${OUTPUT_PATH}/"experiment_${EXPERIMENT_NAME}_$current_time"/train_G.log
