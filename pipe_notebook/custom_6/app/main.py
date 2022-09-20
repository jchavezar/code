
from google.cloud import storage
from fastapi import Request, FastAPI
import json
import os
import pickle
import argparse
import sys

app = FastAPI()

x=os.environ['AIP_STORAGE_URI']
print(f'[INFO] ------ {x}', file=sys.stderr)

# Loading Model File

file_name = 'model.pkl'
client = storage.Client(project=os.environ['PROJECT_ID'])
with open(file_name, "wb") as model:
    client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/{file_name}", model
    )
with open(file_name, 'rb') as file:
    model = pickle.load(file)

# Webserver methods

@app.get('/')
def get_root():
    return {'message': 'Welcome to Breast Cancer Prediction'}
@app.get('/health_check')
def health():
    return 200
if os.environ.get('AIP_PREDICT_ROUTE') is not None:
    method = os.environ['AIP_PREDICT_ROUTE']
else:
    method = '/predict'
print(method)
@app.post(method)
async def predict(request: Request):
    print("----------------- PREDICTING -----------------")
    body = await request.json()
    instances = body["instances"]
    outputs = model.predict(instances)
    print(f'[INFO] ------ {outputs}, {type(outputs)}', file=sys.stderr)
    response = outputs.tolist()
    print("----------------- OUTPUTS -----------------")
    return {"predictions": response}
