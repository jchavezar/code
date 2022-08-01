import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model-name', default='custom-train', help='The name of your saved model')

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    print('accessing main')
    import utils

    utils.save_model(args)
