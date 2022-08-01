
import argparse
import os
import logging
import dask_cudf
import xgboost as xgb
import pandas as pd
#import pickle
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_source', dest='dataset',
                    type=str,
                    help='Dataset.')
parser.add_argument(
    '--job-dir',
    default=os.getenv('AIP_MODEL_DIR'),
    help='GCS location to export models')
parser.add_argument(
    '--model-name',
    default="custom-train",
    help='The name of your saved model')

args = parser.parse_args()

logging.info(f"Importing dataset {args.dataset}")
df = dask_cudf.read_csv(args.dataset)

logging.info("Cleaning and standarizing dataset")
df = df.dropna()

logging.info(f"Splitting dataset")
df_train, df_eval = df.random_split([0.8, 0.2], random_state=123)

df_train_features= df_train.drop('Cover_Type', axis=1)
df_eval_features= df_eval.drop('Cover_Type', axis=1)

df_train_labels = df_train.pop('Cover_Type')
df_eval_labels = df_eval.pop('Cover_Type')

if __name__ == '__main__':
    import utils

    logging.info("Creating dask cluster")
    cluster = LocalCUDACluster()
    client = Client(cluster)
    
    logging.info(client)
    
    # X and y must be Dask dataframes or arrays
    
    print(xgb.__version__)

    logging.info("Dataset for dask")
    dtrain = xgb.dask.DaskDMatrix(client, df_train_features, df_train_labels)
    
    logging.info("Dataset for dask")
    dvalid = xgb.dask.DaskDMatrix(client, df_eval_features, df_eval_labels)

    logging.info("Training")
    output = xgb.dask.train(
        client,
        {
            "verbosity": 2, 
            "tree_method": "gpu_hist", 
            "objective": "multi:softprob",
            "eval_metric": ["mlogloss"],
            "num_class": 8
        },
        dtrain,
        num_boost_round=4,
        evals=[(dvalid, "valid1")],
        early_stopping_rounds=5
    )
    
    # Saving models and exporting performance metrics
    
    df_eval_metrics = pd.DataFrame(output["history"]["valid1"])
    model = output["booster"]
    best_model = model[: model.best_iteration]
    logging.info(f"Best model: {best_model}")
    temp_dir = "/tmp/xgboost"
    os.mkdir(temp_dir)
    best_model.save_model("{}/{}".format(temp_dir, args.model_name))
    df_eval_metrics.to_json("{}/all_results.json".format(temp_dir))

    utils.save_model(args)
