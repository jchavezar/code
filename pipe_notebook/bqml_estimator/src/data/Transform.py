import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

## Tensors

def df_to_dataset(dataframe, batch_size, target_column, shuffle):
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