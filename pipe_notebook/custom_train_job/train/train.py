
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model_uri = os.environ['AIP_STORAGE_URI']

def build_model(train_data):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    
    return model

def train_model(train_data, train_labels, epochs: int = 1000):
    
    print('[INFO] ------ Building Model Layers', file=sys.stderr)
    model = build_model(train_data)
    epochs = epochs
    
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    print('[INFO] ------ Iterations / Training', file=sys.stderr)
    early_history = model.fit(train_data, train_labels, 
        epochs=epochs, validation_split = 0.2, 
        callbacks=[early_stop])
    
    print('[INFO] ------ Saving Model', file=sys.stderr)
    model.save(f'{model_uri}/mpg/model')

    return model
