
from google.cloud import storage
import os
import logging

def save_model(args):
    """Saves the model to Google Cloud Storage or local file system
    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    if args.job_dir.startswith(scheme):
        print(f"Reading input job_dir: {args.job_dir}")
        job_dir = args.job_dir.split("/")
        bucket_name = job_dir[2]
        object_prefix = "/".join(job_dir[3:]).rstrip("/")
        print(f"Reading object_prefix: {object_prefix}")

        if object_prefix:
            model_path = '{}/{}'.format(object_prefix, "xgboost")
        else:
            model_path = '{}'.format("xgboost")
            
        print(f"The model path is {model_path}")
        bucket = storage.Client().bucket(bucket_name)    
        local_path = os.path.join("/tmp", "xgboost")
        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(local_file)
        print(f"gs://{bucket_name}/{model_path}")
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")
