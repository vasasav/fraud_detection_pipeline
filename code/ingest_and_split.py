"""
Load the raw transactions, split them into train/validation and save as PARQUET files. Also load labels and
save them as PARQUET file. Typical call is

```
python code/ingest_and_split.py \
  --source_txn_csv="${SCRIPT_DIR}/data/raw/transactions_obf.csv" \
  --destination_txn_train="${TRAIN_TXNS}" \
  --destination_txn_validate="${EVAL_TXNS}" \
  --source_labels_csv="${SCRIPT_DIR}/data/raw/labels_obf.csv" \
  --destination_labels="${DST_LABELS}" \
  --split_hash="${RUN_HASH}" \
  --validate_fraction=0.3 \
```

Relies on duckdb to handle structured data. See `python code/ingest_and_split.py --help` for more information
"""

import duckdb as ddb
import argparse
import numpy as np


#######################

def parse_arguments() -> argparse.Namespace:
    """
    A convinience function that initialized the argument parser and extracts
    the command line arguments
    :return: parsed arguments, args = parser.parse_args()
        which will contain args.source_txn_csv, args.destination_txn_train,
        args.destination_txn_validate, args.split_hash, args.validate_fraction
    """
    # init parses
    parser = argparse.ArgumentParser(
        description="Load the test exercise CSV, split it into eval/train and save as PARQUET"
    )
    #
    parser.add_argument(
        '--source_txn_csv',
        type=str,
        required=True,
        help='Path to CSV with transactions e.g. `transactions_obf.csv`'
    )
    #
    parser.add_argument(
        '--destination_txn_train',
        type=str,
        required=True,
        help='Path into which the ingested parquet file will be loaded. for training'
    )
    #
    parser.add_argument(
        '--destination_txn_validate',
        type=str,
        required=True,
        help='Path into which the ingested parquet file will be loaded. for validation'
    )
    #
    parser.add_argument(
        '--split_hash',
        type=str,
        required=False,
        default="42fish",
        help='Hash string that will be used to split the file into train-test'
    )
    #
    parser.add_argument(
        '--validate_fraction',
        type=float,
        default=0.3,
        help='Fraction of data that will be stored in the validation dataset (0-1), '+\
             'if less than -0.5 only train dataset will be saved'
    )
    #
    parser.add_argument(
        '--source_labels_csv',
        type=str,
        required=True,
        help='Path to the CSV with labels for the transactions'
    )
    #
    parser.add_argument(
        '--destination_labels',
        type=str,
        required=True,
        help='Destination path for the parquet file with the labels'
    )

    # Parse arguments
    args = parser.parse_args()

    return args


################

def load_source_txns(
        con: ddb.duckdb.DuckDBPyConnection,
        source_txn_csv: str,
        split_hash: str,
) -> None:
    """
    Given initialized duck-db connection, and a location of a CSV file with transactions
    data, load this data into the connection, call the loaded table `txns_table_name`
    append a row number based on hash - it will be used to randomly split data later.

    The loaded table will contain data with the following schema
    'transactionTime': 'TIMESTAMP',
    'eventId': 'VARCHAR',
    'accountNumber': 'VARCHAR',
    'merchantId': 'VARCHAR',
    'mcc': 'VARCHAR',
    'merchantCountry': 'VARCHAR',
    'merchantZip': 'VARCHAR',
    'posEntryMode': 'VARCHAR',
    'transactionAmount': 'DOUBLE',
    'availableCash': 'DOUBLE'
    'random_row_order': 'INT'

    :param con: initialized duckdb connection
    :param source_txn_csv: path to the CSV with source transactions
    :param split_hash: hash string that will be used in splitting
    :return:
    """

    # load
    con.execute(f"""
        CREATE TABLE txns AS
        SELECT 
            *, 
            ROW_NUMBER() OVER(
                ORDER BY HASH(CONCAT(CAST(transactionTime AS STRING), eventId, '{split_hash}'))
            ) AS random_row_order
        FROM read_csv_auto(
            '{source_txn_csv}', 
            types={{ 
                'transactionTime': 'TIMESTAMP', 
                'eventId': 'VARCHAR',
                'accountNumber': 'VARCHAR',
                'merchantId': 'VARCHAR',
                'mcc': 'VARCHAR',
                'merchantCountry': 'VARCHAR',
                'merchantZip': 'VARCHAR',
                'posEntryMode': 'VARCHAR',
                'transactionAmount': 'DOUBLE',
                'availableCash': 'DOUBLE'
            }}
        )
    """)


################

def split_txns_and_save(
        con: ddb.duckdb.DuckDBPyConnection,
        destination_txn_train: str,
        destination_txn_validate: str,
        validate_fraction: float,
) -> None:
    """
    Given an initialized DB connection with loaded transactions table (see `load_source_txns`)
    split the transactions, with `validate_fraction` going for validation and the rest for training
    then save the two resultant tables as two parquet files.

    :param con: initialized duckdb connection
    :param destination_txn_train: destination into which the train part of transactions should be saved
    :param destination_txn_validate: ditto validation part
    :param validate_fraction: fraction (0..1) of the rows that will go towards validation
    :return:
    """


    row_count = con.execute(
        f'SELECT MAX(random_row_order) AS row_count FROM txns'
    ).df().row_count.iloc[0]
    print(f'Loaded {row_count} rows')

    if validate_fraction > -0.5:
        print(f'Splitting data into train-validate...')
        assert validate_fraction > 0.0 and validate_fraction < 1.0
        to_validate_row_count = np.floor(row_count * validate_fraction).astype(int)
        print(
            f'Rows count selected for validation {to_validate_row_count} ' + \
            f'({to_validate_row_count / row_count * 100:.1f}%)'
        )
        #
        print(f'Saving validation transactions ({destination_txn_validate}) ... ')
        # Save the table as a Parquet file
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE(random_row_order) FROM txns WHERE random_row_order<={to_validate_row_count}
            ) TO '{destination_txn_validate}' (FORMAT PARQUET)
            """
        )
        #
        print(f'Saving train/test transactions ({destination_txn_train}) ... ')
        # Save the table as a Parquet file
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE(random_row_order) FROM txns WHERE random_row_order>{to_validate_row_count}
            ) TO '{destination_txn_train}' (FORMAT PARQUET)
            """
        )

    else:
        print('Saving only the train datadset')

        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE(random_row_order) FROM txns
            ) TO '{destination_txn_train}' (FORMAT PARQUET)
            """
        )


################

def ingest_labels(
        con: ddb.duckdb.DuckDBPyConnection,
        source_labels_csv: str,
        destination_labels: str,
) -> None:
    """
    Given an initialized DB connection load labels table from CSV and then save
    it as parquet. Mostly helps to not to worry about field format later on

    :param con: Initidlized duckdb connector
    :param source_labels_csv: Path to the CSV with the labels for transactions
    :param destination_labels: path to the parquet file that will be used to store the labels
    :return:
    """

    con.execute(f"""
        CREATE TABLE labels AS
        SELECT 
            *
        FROM read_csv_auto(
            '{source_labels_csv}', 
            types={{ 
                'reportedTime': 'TIMESTAMP', 
                'eventId': 'VARCHAR'
            }}
        )
    """)
    #
    con.execute(
        f"""
        COPY (
            SELECT * FROM labels
        ) TO '{destination_labels}' (FORMAT PARQUET)
        """
    )

################

def main(args=None)->None:
    """
    Main entrypoint, can be called by importing the file and calling function
    see `parse_arguments` for which arguments need to be passed
    :return:
    """

    # load command line arguments
    if args is None:
        args = parse_arguments()

    #### initialize db connection
    con = ddb.connect()

    #### load the data from csv into the connection
    # also add a row by which we will be able to train-validate randomize
    print('Loading transactions...')
    load_source_txns(
        con=con,
        source_txn_csv=args.source_txn_csv,
        split_hash=args.split_hash
    )

    print('Splitting and saving ingested transactions...')
    # split the transactions and save them separately
    split_txns_and_save(
        con=con,
        destination_txn_train=args.destination_txn_train,
        destination_txn_validate=args.destination_txn_validate,
        validate_fraction=args.validate_fraction,
    )

    print(f'Ingesting the labels ({args.destination_labels})...')
    ingest_labels(
        con=con,
        source_labels_csv=args.source_labels_csv,
        destination_labels=args.destination_labels
    )

    con.close()

    print('Done')

#######################

if __name__ == '__main__':
    main()
