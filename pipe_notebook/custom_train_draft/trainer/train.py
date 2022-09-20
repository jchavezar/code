
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_uri', dest='dataset_uri', help='testing')
args = parser.parse_args()

print(args.dataset_uri)

normed_train_data = pd.read_csv(f"{args.dataset_uri}.train_data.csv")
normed_test_data = pd.read_csv(f"{args.dataset_uri}.test_data.csv")

print(normed_train_data)
