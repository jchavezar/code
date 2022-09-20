
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from custom_draft import ctx

def build_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    
    return model

def train_model(train_data, train_labels, epochs: int = 1000):
    model = build_model(train_data)
    epochs = epochs
    
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    early_history = model.fit(train_data, train_labels, 
        epochs=epochs, validation_split = 0.2, 
        callbacks=[early_stop])
    
    model.save(f'{ctx.MODEL_URI}/mpg/model')

    return model
