
from fastapi import Request, FastAPI
import json
import os
import sys
import tensorflow as tf

app = FastAPI()

x=os.environ['AIP_STORAGE_URI']
print(f'[INFO] ------ {x}', file=sys.stderr)

# Loading Model File

model = tf.keras.models.load_model(os.environ['AIP_STORAGE_URI'])

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
    #columns = ["bounces", "channelGrouping", "country", "deviceCategory", "latest_ecommerce_progress", "medium", "pageviews", "source", "time_on_site"]
    #print(instances, file=sys.stderr)
    #zip_iterator = zip(columns, instances)
    #print(zip_iterator, file=sys.stderr)
    #sample = dict(zip_iterator)
    #print(sample, file=sys.stderr)
    outputs = []
    for i in instances:
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in i.items()}
        predictions = model.predict(input_dict)
        prob = tf.nn.sigmoid(predictions[0])
        outputs.append(round(prob.numpy()[0]))
        print(f'[INFO] ------ {outputs}, {type(outputs)}', file=sys.stderr)
        print("----------------- OUTPUTS -----------------")
        return {"predictions": outputs}
