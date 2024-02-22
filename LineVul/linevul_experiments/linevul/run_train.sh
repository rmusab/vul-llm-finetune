#/bin/bash


EPOCHS=${EPOCHS:-10}
BLOCK_SIZE=${BLOCK_SIZE:-512}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
MODEL_NAME=${MODEL_NAME:-microsoft\/codebert\-base}
TOKENIZER_NAME=${TOKENIZER_NAME:-microsoft\/codebert\-base}
RANDOM=${RANDOM:-1111}
CHOOSE_BEST_THRESH=${CHOOSE_BEST_THRESH:-False}
OPT_METRIC=${OPT_METRIC:-f1}

echo ${PRETRAINED_PATH}
# PRETRAINED_PATH is unset or set to the empty string
if [ -z "${PRETRAINED_PATH}" ]; then
    python3 linevul_main.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=${TOKENIZER_NAME} --model_name_or_path=${MODEL_NAME} --opt_metric=${OPT_METRIC} --do_train --do_test --choose_best_thresh=${CHOOSE_BEST_THRESH} --epochs ${EPOCHS} --block_size ${BLOCK_SIZE} --output_prob_path ${OUTPUT_PATH} --output_dir ${OUTPUT_PATH} --train_batch_size ${TRAIN_BATCH_SIZE} --eval_batch_size ${EVAL_BATCH_SIZE} --learning_rate 2e-5 --max_grad_norm 1.0 --seed ${RANDOM}
else
	python3 linevul_main.py --output_dir=./saved_models --model_type=roberta --pretrained_path=${PRETRAINED_PATH}  --tokenizer_name=${TOKENIZER_NAME} --opt_metric=${OPT_METRIC} --model_name_or_path=${MODEL_NAME} --do_train --do_test --choose_best_thresh=${CHOOSE_BEST_THRESH} --epochs ${EPOCHS} --block_size ${BLOCK_SIZE} --output_prob_path ${OUTPUT_PATH} --output_dir ${OUTPUT_PATH} --train_batch_size ${TRAIN_BATCH_SIZE} --eval_batch_size ${EVAL_BATCH_SIZE} --learning_rate 2e-5 --max_grad_norm 1.0 --seed ${RANDOM}
fi
