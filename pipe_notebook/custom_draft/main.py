
from fastapi import Request, FastAPI
import tensorflow as tf
import json
import os
import preprocess

app = FastAPI()

model_uri=os.environ['AIP_STORAGE_URI']
print(f'[INFO] ------ {model_uri}', file=sys.stderr)
model = tf.keras.models.load_model(f'{model_uri}/mpg/model')

@app.get('/')
def get_root():
    return {'message': 'Welcome mpg API: miles per gallon prediction'}

@app.get('/health_check')
def health():
    return 200

if os.environ.get('AIP_PREDICT_ROUTE') is not None:
    method = os.environ['AIP_PREDICT_ROUTE']
else:
    method = '/predict'

@app.post(method)
async def predict(request: Request):
    print("----------------- PREDICTING -----------------")
    body = await request.json()
    instances = body["instances"]

    norm_data = preprocess.pred_data_process(instances)
    
    outputs = model.predict(norm_data)
    response = outputs.tolist()
    print("----------------- OUTPUTS -----------------")
    return {"predictions": response}
