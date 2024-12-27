#!/bin/bash

# run train, predict, eval end-to-end


set -x
set -e
set -u

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RUN_HASH=$(date +"%Y%m%d")$(openssl rand -hex 3 | head -c 5)

################### EXTRACT CSVs
# The code below extracts the data from task_details/data-new.zip and stores them under data/raw
# some names are hard-coded due to nature of the input
echo "Extracting CSVs"
unzip -j "${SCRIPT_DIR}/task_details/data-new.zip" *.csv -d "${SCRIPT_DIR}/data/raw"


################### INGEST DATA
# the code initializes a python script that loads the transaction data and splits it into
# two sets by rows. The split is random, controlled by a hash-string. The proportion is 
# specified as 0.7 and 0.3 by default. This will be the train/test and validate sets
# the data will be stored as pyarrow to simplify further work with it
TRAIN_TXNS="${SCRIPT_DIR}/data/train_txns_${RUN_HASH}.parquet"
EVAL_TXNS="${SCRIPT_DIR}/data/eval_txns_${RUN_HASH}.parquet"
LABELS="${SCRIPT_DIR}/data/labels.parquet"
#
echo "Ingesting data into PARQUET"
python ${SCRIPT_DIR}/code/ingest_and_split.py \
  --source_txn_csv="${SCRIPT_DIR}/data/raw/transactions_obf.csv" \
  --destination_txn_train="${TRAIN_TXNS}" \
  --destination_txn_validate="${EVAL_TXNS}" \
  --source_labels_csv="${SCRIPT_DIR}/data/raw/labels_obf.csv" \
  --destination_labels="${LABELS}" \
  --split_hash="${RUN_HASH}" \
  --validate_fraction=0.3 \


################## PREPARE CATEGORICAL FEATURE DICTs

CAT_FEATURE_LIST=("posEntryMode")
CAT_ENC_PATH="${SCRIPT_DIR}/data/categorical_encoding_dict_${RUN_HASH}.json"

python ${SCRIPT_DIR}/code/generate_categorical_encoding_dict.py \
  --source_data_table="${TRAIN_TXNS}" \
  --categorical_feature_list ${CAT_FEATURE_LIST[@]} \
  --destination_categorical_encoding_dict="${CAT_ENC_PATH}"

################## CREATE DATASETS READY FOR TRAIN/EVAL

TRAIN_DATASET="${SCRIPT_DIR}/data/train_dataset_${RUN_HASH}.parquet"
#
python ${SCRIPT_DIR}/code/generate_model_dataset.py \
  --source_txns_table="${TRAIN_TXNS}" \
  --destination_model_dataset="${TRAIN_DATASET}" \
  --source_labels_table="${LABELS}" \
  --source_categorical_encoding_dict="${CAT_ENC_PATH}" \


EVAL_DATASET="${SCRIPT_DIR}/data/eval_dataset_${RUN_HASH}.parquet"
#
python ${SCRIPT_DIR}/code/generate_model_dataset.py \
  --source_txns_table="${EVAL_TXNS}" \
  --destination_model_dataset="${EVAL_DATASET}" \
  --source_labels_table="${LABELS}" \
  --source_categorical_encoding_dict="${CAT_ENC_PATH}" \

################# TRAIN MODEL

MODEL_CONFIG="v0.3"
TRAINED_MODEL="${SCRIPT_DIR}/fraud_flagging_model_${MODEL_CONFIG}_${RUN_HASH}.json"

python ${SCRIPT_DIR}/code/train_model.py \
  --source_dataset="${TRAIN_DATASET}" \
  --model_config="${MODEL_CONFIG}" \
  --destination_model="${TRAINED_MODEL}"

################ GENERATE PREDICTIONS

PREDICTIONS="${SCRIPT_DIR}/predictions_${RUN_HASH}.parquet"

python ${SCRIPT_DIR}/code/predict.py \
  --source_dataset="${EVAL_DATASET}" \
  --source_model="${TRAINED_MODEL}" \
  --destination_predictions="${PREDICTIONS}"

############### Evaluate prediction results

EVALUATE_SUMMARY="${SCRIPT_DIR}/evaluation_${RUN_HASH}.json"
MONTHLY_INVESTIGATIVE_CAPACITY=120
ITERATION_COUNT=300

python ${SCRIPT_DIR}/code/evaluate_results.py \
  --source_dataset="${EVAL_DATASET}" \
  --source_predictions="${PREDICTIONS}" \
  --monthly_investigative_capacity=${MONTHLY_INVESTIGATIVE_CAPACITY} \
  --bootstrap_iteration_count=${ITERATION_COUNT} \
  --destination_results=${EVALUATE_SUMMARY} \
  --source_txns="${EVAL_TXNS}"


