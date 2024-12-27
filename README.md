# Machine learning for reduction of fraud losses: technical details

## Running the analysis

End-to-end run can be triggered from `run_end_to_end.sh`. It will:
1. Split the data set into train/evaluate (set `--validate_fraction=-1` to only save train)
2. Ingest data into PARQUET
3. Build dictionaries for the categorical features
4. Prepare datasets that can be used for model training/predictions
5. Train Model
6. Generate predictions
7. Save results

The main code is saved in `./code/*.py` each filed contains a header with description


## Model choice & related information

The native environment for this is Linux enivronment with Python 3.10. Given correct setup, it should run in Windows, but the orchestration script will need changing (`run_end_to_end.sh`). Python scripts will work fine. To aid setup, the key important libraries are listed in `requirements.txt` (install with `pip install -r requirements.txt`).

Model chosen as a core is CatBoost. It is open-source, performance and can save results into JSON. Gradient-boosting algorithm is well-suited for the tabular data. Initial motivation was to also use it's ability to handle categorical data natively, but in the end I decided to generate dictionaries as part of the run, to make the project more portable.

No hyperparameter tuning has been performed - the focus was on achitecture of the pipeline. Four configurations of model have been prepared:

* `v0.0` - model with just two iterations. usually performs badly. Good for sanity checks
* `v0.1` - model on defaults
* `v0.2` - model with up-weighted fraud transactions during train-time
* `v0.3` - same as before, but fraud transactions are additionally up-weighted by `transactionAmount` in order to drive further reduction in the amount lost to fraud.

Train-evaluation split is done accorss all months. The other option would have been to train on earlier months, and predict on later months, however only 13 months of data were given, and preliminary investigation did show quite a strong seasonal component of fraud. This is why `txn_month` is one of the model's features. For the same reason it is necessary to have full 12 months in the training window of the classifer.


## Artefacts generated as a result of a run

* `./data/train_txns_*.parquet` - transactions that will be used for training
* `./data/eval_txns_*.parquet` - transactions that will be used for evaluation
* `./data/labels.parquet` - labels loaded from the raw data 
* `./data/categorical_encoding_dict_*.json` - dictionary with frequency encoding for the categorical features that will be used in the model
* `./data/train_dataset_*.parquet` - dataset that will be used for training (transactions and labels are joined)
* `./data/eval_dataset_*.parquet` - dataset that will be used for evaluation
* `./fraud_flagging_model_v*.json` - trained model 
* `./predictions_*.parquet` - predictions for the eval dataset
* `./evaluation_*.json` - evaluation metrics (e.g. amount of fraud prevented)

Each run is done using a specially-generated run-id (e.g. `20240824ddc70`). The logic is defined in `run_end_to_end.sh`. The run-id is used to both label the files, as well as to split data into train and validate datasets.

## Evaluation methodology

Model is trained on train data. Predictions are generated for the evaluation dataset. The predictions are then used to rank the transactions in terms of fraud risk. Lets say the investigative capacity for evaluation dataset is 120. Then 120 highest-scored transactions are selected each month. Those would have been investigated. Any remaining fraud transactions would be fraud loss. 

As a **baseline**, I simulate and alternative investigations strategy, where one selects the 120 transactions for investigation based on random choice. This is done each month. Again, whatever is not selected, and is fraud, becomes fraud loss. The process is repeated many times (e.g. 300) to get a representative average. This is referred to as *'bootstrapping'*.

As a result, I have monthly fraud loss one gets using a classifier and when one selects transactions randomly. Classifier usually has much lower fraud loss, unless it is v0.0, which is made deliberatly bad. There the perfromance is usually on par with random choice.

## Model comparison

The framework is optimized for individual runs rather than repeated runs. Below I will give several runs and the key results for them. These will be used to extract the metrics that will go into the report

Train/validate split: 0.7/0.3
Investigative capacity for validate: 0.3*400=120


| run-id        | model   | full fraud in evaluation set | fraud prevented          | baseline fraud prevented   | ratio of fraud prevented compared to baseline |
|---------------|---------|------------------------------|--------------------------|----------------------------|-----------------------------------------------|
| 202408247b82a | v0.0    | 32533.4                      | 3153.0                   | 1553.1                     | 2.0                                           |
| 2024082427365 | v0.0    | 34776.0                      | 381.1                    | 1435.6                     | 0.27                                          |
| 202408241865b | v0.0    | 28585.4                      | 585.9                    | 1354.5                     | 0.43                                          |
| 2024082478349 | v0.0    | 31624.0                      | 206.5                    | 1372.4                     | 0.15                                          |
| 2024082478349 | v0.0    | 31678.0                      | 3088.9                   | 1662.7                     | 1.85                                          |
| 20240824a9d6c | v0.0    | 26879.9                      | 2408.3                   | 1118.0                     | 2.15                                          |
| 20240824a3c65 | v0.0    | 30469.0                      | 4852.0                   | 1288.4                     | 3.77                                          |
| 20240824843f3 | v0.0    | 42673.1                      | 3346.5                   | 1847.3                     | 1.81                                          |
| 202408241da78 | v0.0    | 32420.6                      | 889.9                    | 1384.3                     | 0.64                                          |
| 2024082469bd1 | v0.0    | 33654.4                      | 486.4                    | 1478.8                     | 0.33                                          |
|---------------|---------|------------------------------|--------------------------|----------------------------|-----------------------------------------------|
| 20240824b4717 | v0.1    | 21563.1                      | 17322.9                  | 927.0                      | 18.70                                         |
| 2024082428f81 | v0.1    | 27037.1                      | 20827.9                  | 1225.7                     | 16.99                                         |
| 202408240e5cb | v0.1    | 33156.4                      | 14283.7                  | 1435.8                     | 9.94                                          |
| 2024082447691 | v0.1    | 33017.3                      | 27124.5                  | 1539.3                     | 17.60                                         |
| 20240824cff70 | v0.1    | 30755.6                      | 19236.8                  | 1418.6                     | 13.56                                         |
| 202408248d681 | v0.1    | 31628.2                      | 17106.3                  | 1220.7                     | 14.01                                         |
| 20240824ab339 | v0.1    | 23318.4                      | 19300.0                  | 1024.8                     | 18.83                                         |
| 20240824e5627 | v0.1    | 32339.1                      | 16109.6                  | 1439.2                     | 11.19                                         |
| 202408241d553 | v0.1    | 40064.3                      | 26228.6                  | 1795.2                     | 14.6                                          |
| 20240824bcdd9 | v0.1    | 23661.1                      | 22617.1                  | 1044.0                     | 16.46                                         |
|---------------|---------|------------------------------|--------------------------|----------------------------|-----------------------------------------------|
| 20240824ca7f4 | v0.2    | 38743.7                      | 19552.1                  | 1725.2                     | 11.33                                         |
| 20240824c1c8f | v0.2    | 32498.1                      | 18453.3                  | 1275.5                     | 14.47                                         |
| 20240824d8243 | v0.2    | 27156.6                      | 18128.9                  | 1124.5                     | 16.12                                         |
| 20240824b2e8c | v0.2    | 39005.1                      | 24911.5                  | 1628.6                     | 15.30                                         |
| 202408249c841 | v0.2    | 28840.8                      | 21113.8                  | 1269.6                     | 16.63                                         |
| 20240824ae67b | v0.2    | 33114.3                      | 19097.9                  | 1489.0                     | 12.83                                         |
| 20240824d6c48 | v0.2    | 20798.8                      | 16517.8                  | 892.7                      | 18.50                                         |
| 2024082435e7d | v0.2    | 33775.7                      | 20529.9                  | 1566.4                     | 13.11                                         |
| 202408240e259 | v0.2    | 48969.7                      | 30410.9                  | 2007.9                     | 15.15                                         |
| 20240824a5a41 | v0.2    | 24488.5                      | 15966.1                  | 1056.0                     | 15.12                                         |
|---------------|---------|------------------------------|--------------------------|----------------------------|-----------------------------------------------|
| 2024082446cfa | v0.3    | 27771.7                      | 20142.6                  | 1141.6                     | 17.64                                         |
| 2024082464ce7 | v0.3    | 29158.3                      | 19070.6                  | 1161.7                     | 16.41                                         |
| 20240824e6a04 | v0.3    | 37334.1                      | 20297.4                  | 1653.1                     | 12.28                                         |
| 2024082427544 | v0.3    | 35578.9                      | 22882.8                  | 1495.4                     | 15.30                                         |
| 2024082443fcd | v0.3    | 32246.4                      | 27490.1                  | 1415.1                     | 19.43                                         |
| 202408240b9d4 | v0.3    | 29455.9                      | 21486.8                  | 1194.7                     | 17.98                                         |
| 202408246e1c8 | v0.3    | 30019.5                      | 23243.7                  | 1384.2                     | 16.79                                         |
| 20240824dba2f | v0.3    | 26377.8                      | 17504.6                  | 1173.0                     | 14.92                                         |
| 2024082436aa9 | v0.3    | 33557.7                      | 24228.8                  | 1397.8                     | 17.33                                         |
| 2024082496e46 | v0.3    | 31151.9                      | 21388.8                  | 1341.5                     | 15.94                                         |

The ratio metric is easiest to compare, since it compares baseline with model on the same split of train and eval.

| model | ratio of fraud prevented   | standard diviation |
|-------|----------------------------|--------------------|
| v0.0  | 1.3                        | 1.1                |
| v0.1  | 15.2                       | 2.9                |
| v0.2  | 14.9                       | 2.0                |
| v0.3  | 16.4                       | 1.9                |

The performance of the models v0.1...v0.3 is not too different. v0.3 could be a better model, and it would align with how it is configured, but results are too close. Certainly, one can claim about 15x improvement in the fraud prevented using the either of the three models. 

It is also interesting to check which features brought largest benefit. For this we fix the train-validation split and add feautures in order of which gives largest increase in performance this time. Did not have time to dive too deep, but it seems that even the model with just the transactionAmount already can do pretty well. Indeed early histograms did suggest that fraud tends to be at higher transaction amounts.

## Final metrics

Will choose v0.3 as the main model. 

| run-id        | model   | fraud                        | txn amount  |
|---------------|---------|------------------------------|-------------|
| 2024082446cfa | v0.3    | 27771.7                      | 1854808.6   |
| 2024082464ce7 | v0.3    | 29158.3                      | 1899095.8   |
| 20240824e6a04 | v0.3    | 37334.1                      | 1912734.2   |
| 2024082427544 | v0.3    | 35578.9                      | 1901884.0   |
| 2024082443fcd | v0.3    | 32246.4                      | 1902037.7   |
| 202408240b9d4 | v0.3    | 29455.9                      | 1927208.6   |
| 202408246e1c8 | v0.3    | 30019.5                      | 1934028.8   |
| 20240824dba2f | v0.3    | 26377.8                      | 1877014.4   |
| 2024082436aa9 | v0.3    | 33557.7                      | 1961260.0   |
| 2024082496e46 | v0.3    | 31151.9                      | 1900599.0   |

proportion of all fraud compared to txns: 1.64% +-0.16%. With random investigations we can reduce it to 1.57% +- 0.16%.
Another way to look at this. Random invetigations reduce fraud loss by 4.3% +- 0.2%. Using v0.3 reduces fraud loss by 70% +- 8%


