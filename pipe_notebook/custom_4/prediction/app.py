
import os
import sys
from flask import Flask, request, Response, jsonify

app = Flask(__name__)

@app.route(os.environ['AIP_HEALTH_ROUTE'])
def health():
    status_code =  Response(status=200)
    print('[INFO] ------ Testing', file=sys.stderr)
    return status_code

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['GET', 'POST'])
def prediction():
    print('[INFO] ------ Prediction Entry', file=sys.stderr)
    return jsonify({"text": "text"})
