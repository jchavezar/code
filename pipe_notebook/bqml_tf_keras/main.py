import os
import warnings
import argparse
from src import Estimator
warnings.filterwarnings('ignore')

TRAIN_START_DATE = "20160801"
TRAIN_END_DATE = "20170430"

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='batch size to build tf.data.Dataset')
    parser.add_argument(
        '--num_neurons',
        default=32,
        type=int,
        help='The number of neural neurons in the first layer'
        )
    parser.add_argument(
        '--learning_rate',
        type=int,
        help='number of units in the first hidden layer')
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='The number of training iterations over the entire dataset'
        )
    parser.add_argument(
        '--label_column',
        default='will_buy_on_return_visit',
        type=str,
        help='The column to be predicted'
        )
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    VertexTF = Estimator.VertexTF(
        project_id=os.environ['CLOUD_ML_PROJECT_ID'],
        epochs=args.epochs
        )
    train, val, test = VertexTF.query(start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE)
    train_ds, val_ds, test_ds = VertexTF.preprocessing(target_column=args.label_column)
    model = VertexTF.create_model(nn_input=args.num_neurons, lr=args.learning_rate)
    print(train_ds)
    #history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds)

    #hp_metric = history.history['val_accuracy'][-1]
    #print(hp_metric)

if __name__ == "__main__":
    main()
