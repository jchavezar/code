
from custom_draft import ctx

## Data Cleaning and Normalizating, exporting statistics.

def train_pre_process(dataset):
    import pandas as pd

    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    
    train_stats = train_dataset.describe()
    train_stats.pop('MPG')
    train_stats = train_stats.transpose()
    train_stats.to_csv(f'{ctx.MODEL_URI}/mpg/stats.csv')
    train_stats.to_csv('stats_2.csv')
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    
    def norm(x):
        return (x - train_stats['mean'])/train_stats['std']
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    return normed_train_data, train_labels, normed_test_data, test_labels

## Using training statistics to equals normalization.

def pred_data_process(data: list):
    import pandas as pd
    
    column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    dataset = pd.DataFrame([data], columns=column_names)

    dataset = dataset.dropna()

    if (dataset['Origin'] == 1).any():
        dataset = dataset.drop(columns=['Origin'])
        dataset['Europe'] = 0
        dataset['Japan'] = 0
        dataset['USA'] = 1

    elif (dataset['Origin'].any == 2).any():
        dataset = dataset.drop(columns=['Origin'])
        dataset['Europe'] = 1
        dataset['Japan'] = 0
        dataset['USA'] = 0

    elif (dataset['Origin'] == 3).any():
        dataset = dataset.drop(columns=['Origin'])
        dataset['Europe'] = 0
        dataset['Japan'] = 1
        dataset['USA'] = 0

    ## Train stats

    train_stats = pd.read_csv('stats.csv', index_col=[0])
    
    def norm(x):
        return (x - train_stats['mean'])/train_stats['std']
    normed_data = norm(dataset)

    return normed_data
