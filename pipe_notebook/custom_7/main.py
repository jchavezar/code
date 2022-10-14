import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import bigquery 
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

### Bigquery Definition

query = """
WITH all_visitor_stats AS (
SELECT
  fullvisitorid,
  IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit
  FROM `data-to-insights.ecommerce.web_analytics`
  GROUP BY fullvisitorid
)
# add in new features
SELECT * EXCEPT(unique_session_id) FROM (
  SELECT
      CONCAT(fullvisitorid, CAST(visitId AS STRING)) AS unique_session_id,
      # labels
      will_buy_on_return_visit,
      MAX(CAST(h.eCommerceAction.action_type AS INT64)) AS latest_ecommerce_progress,
      # behavior on the site
      IFNULL(totals.bounces, 0) AS bounces,
      IFNULL(totals.timeOnSite, 0) AS time_on_site,
      IFNULL(totals.pageviews, 0) AS pageviews,
      # where the visitor came from
      trafficSource.source,
      trafficSource.medium,
      channelGrouping,
      # mobile or desktop
      device.deviceCategory,
      # geographic
      IFNULL(geoNetwork.country, "") AS country
  FROM `data-to-insights.ecommerce.web_analytics`,
     UNNEST(hits) AS h
    JOIN all_visitor_stats USING(fullvisitorid)
  WHERE 1=1
    # only predict for new visits
    AND totals.newVisits = 1
    AND date BETWEEN '20160801' AND '201b70430' # train 9 months
  GROUP BY
  unique_session_id,
  will_buy_on_return_visit,
  bounces,
  time_on_site,
  totals.pageviews,
  trafficSource.source,
  trafficSource.medium,
  channelGrouping,
  device.deviceCategory,
  country)
"""

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=256,
        type=int,
        help='batch size to build tf.data.Dataset')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='learning rate')
    parser.add_argument(
        '--num_neurons',
        default=32,
        type=int,
        help='number of units in the first hidden layer')
    parser.add_argument(
        '--label_column',
        default='will_buy_on_return_visit',
        type=str,
        help='The column to predict (label/target)')
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='Numbber of epochs for the training; complete pass over dataset')
    args = parser.parse_args()
    return args

def preprocessing(batch_size: int, target_column: str):
  '''Split and Transform data into tf.Dataset, shuffles + batch'''
  
  client = bigquery.Client(project=os.environ['CLOUD_ML_PROJECT_ID'])
  df = client.query(query).to_dataframe()
  train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])


  ## Transforming data to tf.data.Dataset (Multidimmension)
  
  def df_to_dataset(dataframe, batch_size: str, shuffle=True):
    for column in dataframe.columns:
      if dataframe[column].dtype == 'Int64':
        dataframe[column] = dataframe[column].astype(np.int64)
      df = dataframe.copy()
      labels = df.pop(target_column)
      df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
      ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
      if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
      ds = ds.batch(batch_size)
      ds = ds.prefetch(batch_size)
    return ds

  ## Tensorflow Normalization

  def get_normalization_layer(name, dataset):
    
    normalizer = layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])    
    normalizer.adapt(feature_ds)
    
    return normalizer

  ## Encoding for categorical data

  def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):

    if dtype == 'string':
      index = layers.StringLookup(max_tokens=max_tokens)
    else:
      index = layers.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))
    
  train_ds = df_to_dataset(train, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
  
  all_inputs = []
  encoded_features = []

  # Numerical features.
  cat_columns = [i for i in df if df[i].dtypes == 'object' and i != target_column]
  num_columns = [i for i in df if df[i].dtypes == 'int64' and i != target_column]

  for header in num_columns:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

  for header in cat_columns:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(
      name=header,
      dataset=train_ds,
      dtype='string',
      max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)

  return all_inputs, encoded_features, train_ds, val_ds, test_ds

def create_model(all_inputs, encoded_features, nn_input: int, lr: float):
  '''Train model with TF+Keras'''
  all_features = tf.keras.layers.concatenate(encoded_features)
  x = tf.keras.layers.Dense(nn_input, activation="relu")(all_features)
  x = tf.keras.layers.Dropout(0.5)(x)
  output = tf.keras.layers.Dense(1)(x)
  
  model = tf.keras.Model(all_inputs, output)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"])
  return model  

def main():
    args = get_args()
    all_inputs, encoded_features, train_ds, val_ds, test_ds = preprocessing(args.batch_size, args.label_column)
    model = create_model(all_inputs, encoded_features, args.num_neurons, args.learning_rate)
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds)

if __name__ == "__main__":
    main()
