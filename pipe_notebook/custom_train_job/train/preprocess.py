
import os
import sys
import pandas as pd


data_uri = os.environ['AIP_STORAGE_URI']

## Data Cleaning and Normalizating, exporting statistics.

def train_pre_process(dataset):

    # Cleaning data and doing transformations

    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    
    train_stats = train_dataset.describe()
    train_stats.pop('MPG')
    train_stats = train_stats.transpose()

    # Storing stats for transformations in Google Cloud Storage

    train_stats.to_csv(f'{data_uri}/mpg/stats.csv')
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    
    # Standarization (Z-Score Normalization)

    def norm(x):
        return (x - train_stats['mean'])/train_stats['std']
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    return normed_train_data, train_labels, normed_test_data, test_labels

## Using training statistics to equals standarization.

def pred_data_process(data: list):
    column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    region_list = ['USA', 'Europe', 'Japan']
    
    dataset = pd.DataFrame([data], columns=column_names)
    dataset = dataset.dropna()

    for data in region_list:
        if dataset['Origin'][0] == data:
            dataset[data] = 1
        else: dataset[data] = 0
    
    dataset = dataset.drop(columns=['Origin'])

    ## Train stats
    train_stats = pd.read_csv(f'{data_uri}/mpg/stats.csv', index_col=[0])
    
    def norm(x):
        return (x - train_stats['mean'])/train_stats['std']
    
    return norm(dataset)
