# Single, Mirror and Multi-Machine Distributed Training for Boston Housing

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os
import sys
tfds.disable_progress_bar()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))

def make_dataset():

  # Scaling Boston Housing data features
  def scale(feature):
    max = np.max(feature)
    feature = (feature / max).astype(np.float)
    return feature, max

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
  )
  params = []
  for _ in range(13):
    x_train[_], max = scale(x_train[_])
    x_test[_], _ = scale(x_test[_])
    params.append(max)

  # store the normalization (max) value for each feature
  with tf.io.gfile.GFile('/tmp/param.txt', 'w') as f:
    f.write(str(params))
  return (x_train, y_train), (x_test, y_test)


# Build the Keras model
def build_and_compile_dnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu', input_shape=(13,)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='linear')
  ])
  model.compile(
      loss='mse',
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01))
  return model

NUM_WORKERS = 1
BATCH_SIZE = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

model = build_and_compile_dnn_model()

# Train the model
(x_train, y_train), (x_test, y_test) = make_dataset()
model.fit(x_train, y_train, epochs=5, batch_size=GLOBAL_BATCH_SIZE)
model.save(.)
