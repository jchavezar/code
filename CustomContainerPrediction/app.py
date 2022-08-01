import json
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import simplejson
from flask import Flask, request, Response
import xgboost as xgb

# Set environnment variables
MODEL_PATH = 'model.json'
MODEL_PATH = os.environ["MODEL_PATH"]

# Loading model
print("Loading model from: {}".format(MODEL_PATH))
model = xgb.Booster(model_file=MODEL_PATH)

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
	_features = ['Id','Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                          'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
                          'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
                          'Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19', 
                          'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
                          'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
	
	data = request.get_json(silent=True, force=True)
	data = data["instances"]
	df = pd.DataFrame(data, columns=_features)
	dmf = xgb.DMatrix(df)
	response = pd.DataFrame(model.predict(dmf))
	response = response.idxmax(axis=1)[0]
	print(f"Cover Type is:  {response}")
	return simplejson.dumps({"Cover Type": str(response)})

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=8080)