"""
Generate dictionaries for converting categorical features into numerical features. The encoding used is frequency
encoding.

For example for feature_values ['a', 'b', 'a', 'c', 'b', 'a']

    The encoding will be: {'a': 3/6, 'b': (3+2)/6, 'c': (3+2+1)/6}
    The encoding is normalized cumulative counts of value occurrence

Typical call is:

python code/generate_categorical_encoding_dict.py \
  --source_data_table="${TRAIN_TXNS}" \
  --categorical_feature_list ${CAT_FEATURE_LIST[@]} \
  --destination_categorical_encoding_dict="${CAT_ENC_PATH}"

Where
    `source_data_table` is the source data saved as PARQUET
    `categorical_feature_list` is the list of categorical features which should be encoded
    `destination_categorical_encoding_dict` is the destination into which a dictionary of dictionaries
        will be saved
"""

import argparse
import numpy as np
import json
import typing as tp
import duckdb as ddb
import pandas as pd

def parse_arguments() ->  argparse.Namespace:
    """
    A convinience function that initialized the argument parser and extracts
    the command line arguments
    :return: parsed arguments, args = parser.parse_args()
        which will contain args.source_data_table, args.categorical_feature_list
        args.destination_categorical_encoding_dict
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate frequency-coded encoding for categorical features")

    # Add arguments
    parser.add_argument(
        '--source_data_table',
        type=str,
        required=True,
        help='Path to the source data table (PARQUET file)'
    )
    #
    parser.add_argument(
        '--categorical_feature_list',
        type=str,
        nargs='+',
        required=True,
        help='List of categorical features (fields in the `source_data_table`)'
    )
    #
    parser.add_argument(
        '--destination_categorical_encoding_dict',
        type=str,
        required=True,
        help='Path to save the categorical encoding dictionary. The encoding will be saved as JSON dictionary'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

##########################

def frequency_encode_feature(
        con: ddb.duckdb.DuckDBPyConnection,
        feature_name: str
)->tp.Dict[str, float]:
    """
    Generate a dictionary with frequency encoding for a categorical table. The input
    is an initialized DuckDB connector with `data_table` table that contains field `feature_name`.
    The count how many times each value of this feature occurs in the population, end then generate
    a frequency encoding. For example for feature_values ['a', 'b', 'a', 'c', 'b', 'a']

    The encoding will be: {'a': 3/6, 'b': (3+2)/6, 'c': (3+2+1)/6}
    The encoding is normalized cumulative counts of value occurrence

    :param con: initialized duckdb connection
    :param feature_name: name of the feauture to encode (expecting to find it in table `data_table`)
    :return: dictionary with encoding, as described above
    """

    # initially planned to use SQL to do cumulative sum as well, but SUM(...) OVER(...)
    # seems to be using something like float32 by default, which leads to overflows
    # using numpy with float64 to avoid this, hence handling the cumsum in python
    cat_counts_df = con.execute(
        f"""
            SELECT
                IFNULL({feature_name}, 'NULL') AS {feature_name},
                COUNT(*) AS row_count
            FROM data_table
            GROUP BY {feature_name}
            """
    ).df().sort_values('row_count', ascending=False)

    feature_cat_name_arr = cat_counts_df[feature_name].values
    feature_cumul_counts_arr = np.cumsum(cat_counts_df.row_count.values)
    feature_enc_arr = feature_cumul_counts_arr / np.max(feature_cumul_counts_arr)

    freq_enc_dict = {
        name: val for name, val in zip(feature_cat_name_arr, feature_enc_arr)
    }

    return freq_enc_dict

##########################

def main(args=None)->None:
    """
    Main entrypoint, can be called by importing the file and calling function
    see `parse_arguments` for which arguments need to be passed
    :return:
    """

    if args is None:
        args = parse_arguments()

    print('Building dictionaries for encoding the categorical features into numerical values')

    #### initialize db connection
    con = ddb.connect()
    con.execute(f"CREATE TABLE data_table AS SELECT * FROM '{args.source_data_table}'")

    assert len(args.categorical_feature_list) > 0, 'Expecting at least one categorical feature name'

    enc_dict = {}
    for i_fn, fn in enumerate(args.categorical_feature_list):
        print(f'Encoding {fn} ({i_fn+1}/{len(args.categorical_feature_list)})')
        enc_dict[fn] = frequency_encode_feature(con, fn)

    print(f'Saving encoding to {args.destination_categorical_encoding_dict}')

    with open(args.destination_categorical_encoding_dict, 'w') as fh:
        json.dump(enc_dict, fh, indent=4)

    print('Done')
    con.close()

if __name__ == "__main__":
    main()