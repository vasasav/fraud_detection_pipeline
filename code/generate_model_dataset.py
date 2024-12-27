"""
Converts transactions and labels, and encoding files into a dataset that is ready for training
or evaluation. Typicall call is

python code/generate_model_dataset.py \
  --source_txns_table="${TRAIN_TXNS}" \
  --destination_model_dataset="${TRAIN_DATASET}" \
  --source_labels_table="${LABELS}" \
  --source_categorical_encoding_dict="${CAT_ENC_PATH}" \

Where
    `source_txns_table` is the transactions PARQUET file (see ingest_and_split.py)
    `destination_model_dataset` is the desination path for the PARQUET file that will be created
                                    that file will contain only numerical features apart from `eventId`
                                    the special id column, that will remain string
    `source_labels_table`: labels table as PARQUET file (see ingest_and_split.py). This is optional
                                if not passed the corresponding label column will still be created
                                but will be filled with -1 (not 0 or 1)
    `source_categorical_encoding_dict`: JSON dictionary with categorical feature encodings
                                (see `generate_categorical_encoding_dict.py`). Optional
                                but if cateforical columns are selected in the model, and the encoding
                                is not available, then the script will fail with an exception
                                If encoding for some categorical feature is not present in the dict
                                then the script will also fail

The columns that will be used by the model are encoded in this script. For now, decided
not to parametrize this any further as it will increase complexity. Day-2 move might
be to make features selected for the model be specified in some external config file

NOTE! As part of the execution the transactions are loaded fully into the memory, if
the file is too big, break it into smaller sections and run script on them indepently

Typical columns in the dataset are:
    eventId - reserved name (string) for the id of rows
    txn_month - month of the transaction
    txn_day_of_month - day of the transaction in a month
    txn_day_of_week - doy of the week for the transaction
    txn_hour - hour of the transaction
    posEntryMode - The Point Of Sale entry mode
    transactionAmount - The value of the transaction in GBP
    availableCash - The (rounded) amount available to spend prior to the transaction
    is_fraud_flag - 0/1 flag to indicate fraud (will be set to -1 if labels were not provided)

"""

import argparse
import duckdb as ddb
import json
import pandas as pd
import numpy as np

def parse_arguments() ->  argparse.Namespace:
    """

    :return:
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Generate a model dataset from transaction and label tables."
    )

    # Add arguments
    parser.add_argument(
        '--source_txns_table',
        type=str,
        required=True,
        help='Path to the source transactions table (PARQUET)'
    )
    #
    parser.add_argument(
        '--source_labels_table',
        type=str,
        required=False,
        default=None,
        help='Path to the source labels table (PARQUET). If not provided, the label fields will be filled with -1'
    )
    #
    parser.add_argument(
        '--source_categorical_encoding_dict',
        type=str,
        required=False,
        default=None,
        help='Path to the categorical encoding dictionary (see `generate_categorical_encoding_dict.py`)'
    )
    #
    parser.add_argument(
        '--destination_model_dataset',
        type=str,
        required=True,
        help='Path to save the generated model dataset (PARQUET)'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

###################

def main(args=None)->None:
    """
    Main entrypoint, can be called by importing the file and calling function
    see `parse_arguments` for which arguments need to be passed
    :return:
    """

    if args is None:
        args = parse_arguments()

    # name of the id for different transactions
    # too deep of a parameter to expose
    # but don't like to repeatedly hard-code it
    row_id_name = 'eventId'

    #### initialize db connection
    con = ddb.connect()

    print('Loading txns')
    con.execute(f"CREATE TABLE txns AS SELECT * FROM '{args.source_txns_table}'")

    labels_provided = (args.source_labels_table is not None)
    if labels_provided:
        print('Loading labels')
        con.execute(f"CREATE TABLE labels AS SELECT * FROM '{args.source_labels_table}'")
    else:
        con.execute(f"CREATE TABLE labels (reportedTime TIMESTAMP, eventId STRING);")

    enc_dict_provided = (args.source_categorical_encoding_dict is not None)
    if enc_dict_provided:
        print('Loading categorical encodings')

        with open(args.source_categorical_encoding_dict, 'r') as fh:
            cat_enc_dict = json.load(fh)
    else:
        cat_enc_dict = None

    ### create a single table with all features, without handling
    # categorical values
    # this is essentially where the tables that will be used for ML are selected
    print('Preparing key fields')
    con.execute(
        f"""
        CREATE TABLE full_table AS (
            SELECT
                T.{row_id_name},
                T.transactionTime,
                MONTH(T.transactionTime) AS txn_month,
                DAYOFMONTH(T.transactionTime) AS txn_day_of_month,
                DAYOFWEEK(T.transactionTime) AS txn_day_of_week,
                HOUR(T.transactionTime) AS txn_hour,
                T.posEntryMode,
                T.transactionAmount,
                T.availableCash,
                L.reportedTime,
            FROM txns AS T
            LEFT JOIN labels AS L
                ON L.eventId=T.eventId
        )
        """
    )

    ### load the table as a dataframe
    print('Loading the dataframe...')
    prelim_dataset_df = con.execute(
        f"""
        WITH
        core_vw AS (
            SELECT 
                * EXCLUDE(transactionTime, reportedTime),
                CASE
                    WHEN reportedTime IS NULL THEN {'0' if labels_provided else '-1'}
                    ELSE 1
                END AS is_fraud_flag,
            FROM full_table
        )
    
        SELECT
            *
        FROM core_vw
        """
    ).df()
    #
    print(f'Loaded {len(prelim_dataset_df)} rows. Columns {prelim_dataset_df.columns}')

    # handle categorical values
    print(f'Handling the following categorical features: ')
    categorical_feature_list = []
    for col_name in prelim_dataset_df.columns:
        if ((prelim_dataset_df.dtypes[col_name]==type(str)) and (col_name!=row_id_name)):

            if cat_enc_dict is not None and col_name in cat_enc_dict:
                print(f'\t{col_name}')
                categorical_feature_list.append(col_name)
            else:
                raise Exception(f'Categorical encoding for `{col_name}` is missing')

    # apply encoding to the categorical features
    if len(categorical_feature_list) == 0:
        print('No categorical features')
        dataset_df = prelim_dataset_df
    else:
        # all the non-categorical columns
        dataset_df = prelim_dataset_df[[
            col for col in prelim_dataset_df.columns if col not in categorical_feature_list
        ]]

        # assign numberical values instead of categorical ones
        for cat_feat in categorical_feature_list:
            print(f'Mapping `{cat_feat}`')
            replace_func = np.vectorize(cat_enc_dict[cat_feat].get)
            mapped_field_vals = replace_func(prelim_dataset_df[cat_feat].values)

            # handle values that do not occur in the dictionary
            # there should be no such values, if there are
            # the sanest option is to fail loud
            if not np.all(np.isfinite(mapped_field_vals)):
                raise Exception('Some values have not been found in the dictionary!')

            dataset_df = dataset_df.assign(**{cat_feat: mapped_field_vals})

    ### fully loaded, can now save
    print(f'Saving to {args.destination_model_dataset}')
    dataset_df.to_parquet(args.destination_model_dataset)
    print('Done')

    con.close()

if __name__ == "__main__":
    main()