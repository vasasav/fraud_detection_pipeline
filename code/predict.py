"""
Generate predictions for the given dataset and a trained model

Typicall call is:

```
python code/predict.py \
  --source_dataset="${EVAL_DATASET}" \
  --source_model="${TRAINED_MODEL}" \
  --destination_predictions="${PREDICTIONS}"
```

Where:
    `source_dataset` - is the PARQUET file with model data for predictions (see `generate_model_dataset.py`)
    `source_model` - is the JSON-saved model for predictions (see `train_model.py`)
    `destination_predictions` - is the destination for the PARQUET file with predictions
        it will have columns `eventId` and `score`, the latter being the score indiciating the fraud risk
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
        which will contain args.source_dataset, args.source_model
        args.destination_predictions
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate predictions using a trained model")

    # Add arguments
    parser.add_argument(
        '--source_dataset',
        type=str,
        required=True,
        help='Dataset produced by `generate_model_dataset.py`, a PARQUET files with numerical fields for training'
    )
    #
    parser.add_argument(
        '--source_model',
        type=str,
        required=True,
        help='Trained model saved as a JSON (see `train_model.py`)'
    )
    #
    parser.add_argument(
        '--destination_predictions',
        type=str,
        required=True,
        help='Path to save the predictions (PARQUET)'
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

    print(f'Reading {args.source_dataset}')
    full_df = pd.read_parquet(args.source_dataset)
    event_id_col = full_df.pop('eventId')
    #label_col = full_df.pop('is_fraud_flag')

    print(f'Loading {args.source_model}')
    model = cb.CatBoostClassifier().load_model(args.source_model, format='json')

    print('Generating scores')
    model_scores = model.predict_proba(full_df)[:,1]

    print(f'Saving scores to {args.destination_predictions}')
    save_df = pd.DataFrame(
        columns=['eventId'],
        data=event_id_col
    ).assign(score=model_scores)
    print(f'Number of predictions saved {len(save_df)}')
    save_df.to_parquet(args.destination_predictions)

    print('Done')

if __name__ == "__main__":
    main()
