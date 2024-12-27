import duckdb as ddb
import numpy as np

con = ddb.connect()

# Load the Parquet file into DuckDB
con.execute(f"CREATE TABLE txns AS SELECT * FROM 'eval_txns.parquet'")
con.execute(f"CREATE TABLE labels AS SELECT * FROM 'labels.parquet'")

# Execute a query to select 5 rows from the 'transactions' table
res_df = con.execute(
    f"""
    SELECT
        T.transactionTime,
        L.reportedTime,
        T.eventId,
        T.transactionAmount,
        T.availableCash
    FROM txns AS T
    LEFT JOIN labels AS L
        ON L.eventId=T.eventId
    WHERE RANDOM()<0.1
    """
).df()

con.close()

print(f'Labels for {np.mean(res_df.reportedTime==res_df.reportedTime)*100:.3f}%')
print(res_df.query('reportedTime==reportedTime')[['transactionTime', 'reportedTime']])
