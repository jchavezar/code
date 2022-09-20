
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.distributed import wait
from dask import array as da
import xgboost as xgb
import pandas as pd
from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix
import argparse
import time
import utils
import gcsfs
import dask_cudf
import os, json
import subprocess
import pandas as pd
from dask.utils import parse_bytes
import dask_bigquery

parser = argparse.ArgumentParser()
parser.add_argument(
    '--project_id', 
    dest='project_id',
    type=str,
    help='The Project ID')
parser.add_argument(
    '--bq_dataset', 
    dest='bq_dataset',
    type=str,
    help='BigQuery Dataset')
parser.add_argument(
    '--bq_table', 
    dest='bq_table',
    type=str,
    help='BigQuery Table')
parser.add_argument(
    '--job-dir',
    dest='job_dir',
    type=str,
    default=os.getenv('AIP_MODEL_DIR'),
    help='GCS location to export models')
parser.add_argument(
    '--model-name',
    dest='model_name',
    default="custom-train",
    help='The name of your saved model')
parser.add_argument(
    '--num-gpu-per-worker', type=str, help='num of workers',
    default=2)
parser.add_argument(
    '--threads-per-worker', type=str, help='num of threads per worker',
    default=4)
args = parser.parse_args()


def using_quantile_device_dmatrix(
    client: Client,
    project_id,
    dataset_source,
    table, 
    job_dir, 
    model_name):
    
    start_time = time.time()
    print(f"[INFO] ------ Importing '{dataset_source}/{table}' dataset from BigQuery")
    df = dask_bigquery.read_gbq(
        project_id=project_id,
        dataset_id=dataset_source,
        table_id=table)
    df = dask_cudf.from_dask_dataframe(df)
    print(f"[INFO] ------ Import Done")

    print("Cleaning and standarizing dataset")
    df = df.dropna() 

    print(f"[INFO] ------ Splitting dataset")
    df_train, df_eval = df.random_split([0.8, 0.2], random_state=123)
    df_train_features= df_train.drop('Cover_Type', axis=1)
    df_eval_features= df_eval.drop('Cover_Type', axis=1)
    df_train_labels = df_train.pop('Cover_Type')
    df_eval_labels = df_eval.pop('Cover_Type')

    print(xgb.__version__)

    print("[INFO] ------ Dataset for dask")
    dtrain = dxgb.DaskDeviceQuantileDMatrix(client, df_train_features, df_train_labels)
    
    print("[INFO] ------ Dataset for dask")
    dvalid = dxgb.DaskDeviceQuantileDMatrix(client, df_eval_features, df_eval_labels)
    print("[INFO]: ------ QuantileDMatrix is formed in {} seconds ---".format((time.time() - start_time)))

    del df_train_features
    del df_train_labels
    del df_eval_features
    del df_eval_labels
    
    start_time = time.time()
    print("Training")
    output = xgb.dask.train(
        client,
        {
            "verbosity": 2, 
            "tree_method": "gpu_hist", 
            "objective": "multi:softprob",
            "eval_metric": ["mlogloss"],
            "learning_rate": 0.1,
            "gamma": 0.9,
            "subsample": 0.5,
            "max_depth": 9,
            "num_class": 8
        },
        dtrain,
        num_boost_round=10,
        evals=[(dvalid, "valid1")],
        early_stopping_rounds=5
    ) 
    print("[INFO]: ------ Training is completed in {} seconds ---".format((time.time() - start_time)))

    # Saving models and exporting performance metrics
    
    df_eval_metrics = pd.DataFrame(output["history"]["valid1"])
    model = output["booster"]
    best_model = model[: model.best_iteration]
    print(f"[INFO] ------ Best model: {best_model}")
    temp_dir = "/tmp/xgboost"
    os.mkdir(temp_dir)
    print(job_dir)
    best_model.save_model("{}/{}".format(temp_dir, model_name))
    df_eval_metrics.to_json("{}/all_results.json".format(temp_dir))

    utils.save_model(args)

def get_scheduler_info():
    scheduler_ip =  subprocess.check_output(['hostname','--all-ip-addresses'])
    scheduler_ip = scheduler_ip.decode('UTF-8').split()[0]
    scheduler_port = '8786'
    scheduler_uri = '{}:{}'.format(scheduler_ip, scheduler_port)
    return scheduler_ip, scheduler_uri

if __name__ == '__main__':
    print("[INFO] ------ Creating dask cluster")
    
    sched_ip, sched_uri = get_scheduler_info()
    
    print(f"[INFO] ------ Sched_ip and Sched_uri, {sched_ip}, {sched_uri}")

    print("[INFO]: ------ LocalCUDACluster is being formed ")
    
    with LocalCUDACluster(
        ip=sched_ip,
        n_workers=int(args.num_gpu_per_worker), 
        threads_per_worker=int(args.threads_per_worker) 
    ) as cluster:
        with Client(cluster) as client:
            print('[INFO]: ------ Calling main function ')
            using_quantile_device_dmatrix(
                client, 
                project_id=args.project_id,
                dataset_source=args.bq_dataset,
                table=args.bq_table, 
                job_dir=args.job_dir, 
                model_name=args.model_name)
