import argparse
from Data import DataLoader
from time import time
from mlflow import log_metric
import pickle
import numpy as np


def pickle_steps(path, steps):
    with open(path, 'wb+') as f:
        pickle.dump(steps, f)


def main():
    from spn.algorithms.LearningWrappers import learn_cr_enabled, learn_parametric

    print(f'Load dataset {args.d}')
    loader = DataLoader()
    data = loader.load(dataset=args.d, shuffle=True, seed=args.s)[:1000]
    ds_context = loader.context(args.d, cr_enabled=True)
    ds_context.add_domains(data)

    print('Start training of removal enabled SPN')
    start_training = time()
    learn_cr_enabled(data, ds_context, epsilon=args.e, seed=args.s)
    end_training = time()
    t_removal_enabled = end_training - start_training
    print(f'Training duration: {t_removal_enabled}')
    log_metric('tRemovalEnabledTraining', t_removal_enabled)

    ds_context = loader.context(args.d, cr_enabled=False)
    ds_context.add_domains(data)
    print('Start training of SPN')
    rand_gen = np.random.RandomState(args.s)
    start_training = time()
    learn_parametric(data, ds_context, rand_gen=rand_gen)
    end_training = time()
    t_training = end_training - start_training
    print(f'Training duration: {t_training}')
    log_metric('tLearnSPN', t_training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare runtime of training algorithms.')
    parser.add_argument('-d', type=str, required=True, help='Training dataset: msnbc, plants, adult, abalone, wine')
    parser.add_argument('-e', type=float, required=True, help='Epsilon value for Q-k-Means')
    parser.add_argument('-s', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    main()
