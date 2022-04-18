from torch.nn.utils import prune
from models.vgg import VGG
from argparse import ArgumentParser
import random
from utils import *
models = {'VGG': VGG()}

def main():
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--sample_num', default=1000, type=int)
    parser.add_argument('--model_names', default='VGG')
    parser.add_argument('--data_dir', default='./dataset')

    args = parser.parse_args()
    sample_num = args.sample_num
    model_names = args.model_name
    data_dir = args.data_dir

    for model_name in model_names:
        model = models[model_name]
        for _ in range(sample_num):
            amount = random.random()
            pruned_model = prune.random_unstructured(model, 'weight', amount)
            features = get_features_of_pruned_model(pruned_model)
            latency = get_latency_of_pruned_model(pruned_model)

            with open(data_dir+'modeldata.csv', 'a', encoding='utf-8') as f:
                f.write(features, ',', latency)


if __name__ == '__main__':
    main()