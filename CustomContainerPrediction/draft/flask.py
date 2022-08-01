import json
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import simplejson
from flask import Flask, request, Response

# Set environnment variables
#MODEL_PATH = 'catboost_model.cbm'
#MODEL_PATH = os.environ["MODEL_PATH"]

# Loading model
#catboost_model = CatBoostClassifier()
#print("Loading model from: {}".format(MODEL_PATH))
#catboost_model = catboost_model.load_model(MODEL_PATH, format='cbm')

# Creation of the Flask app
app = Flask(__name__)

# Flask route for Liveness checks
@app.route("/isalive")
def isalive():
	print("/isalive request")
	status_code = Response(status=200)
	return status_code

# Flask route for predictions
@app.route('/predict',methods=['GET','POST'])
def prediction():
	return 'hi how are you?'

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=5000)