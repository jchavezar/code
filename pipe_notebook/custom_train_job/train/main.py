
import sys
import preprocess
import train
import pandas as pd
from tensorflow import keras

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t',sep=" ", skipinitialspace=True)

## Clean, Normalize and Split Data

print('[INFO] ------ Preparing Data', file=sys.stderr)
train_data, train_labels, test_data, test_labels = preprocess.train_pre_process(dataset)

## Train model and save it in Google Cloud Storage

print('[INFO] ------ Training Model', file=sys.stderr)
train.train_model(train_data, train_labels)
