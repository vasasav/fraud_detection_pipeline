"""
Train the model and save the result into JSON file. Currently the model is based on CatBoostClassifier.

Typical call is

python code/train_model.py \
  --source_dataset="${TRAIN_DATASET}" \
  --model_config="${MODEL_CONFIG}" \
  --destination_model="${TRAINED_MODEL}"

Where:
    `source_dataset` - is the PARQUET dataset for training (see `generate_model_dataset.py`)
    `model_config` - a string to indicate which model configuration to use. Only a small number
        of conifurations are supported: `v0.0`, `v0.1`, `v0.2`. Invalid confiuration
        will lead to an error
    `destination_model` - path for saving the trained model as a JSON
"""

import argparse
import catboost as cb
import numpy as np
import pandas as pd

#######################

def parse_arguments() ->  argparse.Namespace:
    """
    A convinience function that initialized the argument parser and extracts
    the command line arguments
    :return: parsed arguments, args = parser.parse_args()
        which will contain args.source_dataset, args.model_config
        args.destination_model
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train a model and save the weights.")

    # Add arguments
    parser.add_argument(
        '--source_dataset',
        type=str,
        required=True,
        help='Dataset produced by `generate_model_dataset.py`, a PARQUET files with numerical fields for training'
    )
    #
    parser.add_argument(
        '--model_config',
        type=str,
        required=True,
        help='Model configuration a single string which will be recognized by the script'
    )
    #
    parser.add_argument(
        '--destination_model',
        type=str,
        required=True,
        help='Path to save the trained model (JSON)'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


#######################

def main(args=None)->None:
    """
    Main entrypoint, can be called by importing the file and calling function
    see `parse_arguments` for which arguments need to be passed
    :return:
    """

    if args is None:
        args = parse_arguments()

    # split feautes and labels
    print(f'Reading {args.source_dataset}')
    full_df = pd.read_parquet(args.source_dataset)
    full_df.pop('eventId')
    label_name = 'is_fraud_flag'
    print(f'{len(full_df)} rows loaded')

    # store the order of the features accepted by the model
    model_feature_list = [col for col in full_df.columns if col!=label_name]
    train_X = full_df[model_feature_list]
    train_Y = full_df[label_name].values

    # select model and train
    print('Training the model....')
    if args.model_config == 'v0.0':
        # basic model, deliberatly bad
        print(f'Selected model config is {args.model_config}')
        model = cb.CatBoostClassifier(verbose=False, iterations=2)
        model.fit(train_X, train_Y)

    elif args.model_config == 'v0.1':
        # basic model
        print(f'Selected model config is {args.model_config}')
        model = cb.CatBoostClassifier(verbose=False)
        model.fit(train_X, train_Y)

    elif args.model_config == 'v0.2':
        # model with up-weighted targets
        print(f'Selected model config is {args.model_config}')

        train_target_rate = np.mean(train_Y)
        weight_vec = np.abs(
            (train_Y * (1 / train_target_rate) ) + (1 - train_Y)
        )

        model = cb.CatBoostClassifier(verbose=False)
        model.fit(cb.Pool(train_X, train_Y, weight=weight_vec))

    elif args.model_config == 'v0.3':
        # model with up-weighted targets, and targets additionally weighted by transactionAmount
        print(f'Selected model config is {args.model_config}')

        txn_amount_vec = train_X['transactionAmount'].values
        train_target_rate = np.mean(train_Y)
        weight_vec = np.abs(
            (train_Y * (1/train_target_rate) * (txn_amount_vec/np.max(txn_amount_vec))) + (1 - train_Y)
        )

        model = cb.CatBoostClassifier(verbose=False)
        model.fit(cb.Pool(train_X, train_Y, weight=weight_vec))

    else:
        raise Exception('Model config not supported')

    print('Training completed')
    print(f'Saving model to. {args.destination_model}')

    model.save_model(args.destination_model, format='json')

    print('Done')

if __name__ == "__main__":
    main()
