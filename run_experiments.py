import argparse
from Data import DataLoader
import matplotlib.pyplot as plt
from time import time
from mlflow import log_metric, log_artifact
import pickle
import numpy as np


def pickle_steps(path, steps):
    with open(path, 'wb+') as f:
        pickle.dump(steps, f)


def main():
    from spn.algorithms.Validity import is_valid_cr
    from spn.algorithms.LearningWrappers import learn_cr_enabled
    from spn.algorithms.StructureLearning import unlearn

    print(f'Load dataset {args.d}')
    loader = DataLoader()
    data = loader.load(dataset=args.d, shuffle=True, seed=args.s)[:1000]
    ds_context = loader.context(args.d, cr_enabled=True)
    ds_context.add_domains(data)

    print('Start training of initial SPN')
    start_training = time()
    scrubbed_spn = learn_cr_enabled(data, ds_context, epsilon=args.e, seed=args.s)
    end_training = time()
    print(f'Training duration: {end_training - start_training}')
    plt.show()

    num_delete = min([args.r, data.shape[0]-1])
    t_forgetting_steps = []
    t_retraining_steps = []
    t_forgetting = 0
    t_retraining = 0

    for i in range(num_delete):
        start_forgetting = time()
        scrubbed_spn = unlearn(scrubbed_spn, data[i:], 0, ds_context, epsilon=args.e, seed=args.s)
        end_forgetting = time()
        t_forgetting_steps.append(end_forgetting - start_forgetting)
        log_metric('tUnlearn', end_forgetting - start_forgetting, step=i)
        t_forgetting += end_forgetting - start_forgetting
        assert is_valid_cr(scrubbed_spn)

        start_retraining = time()
        retrained_spn = learn_cr_enabled(data[i+1:], ds_context, epsilon=args.e, seed=args.s)
        end_retraining = time()
        t_retraining_steps.append(end_retraining - start_retraining)
        log_metric('tRetrain', end_retraining - start_retraining, step=i)
        t_retraining += end_retraining - start_retraining
        assert is_valid_cr(retrained_spn)

    log_metric('tUnlearnTotal', t_forgetting)
    log_metric('tRetrainTotal', t_retraining)

    unlearn_path = f'artifacts/unlearn_{args.d.lower()}_{args.e}_{args.s}_{args.r}.pkl'
    pickle_steps(unlearn_path, np.array(t_forgetting_steps))
    retrain_path = f'artifacts/retrain_{args.d.lower()}_{args.e}_{args.s}_{args.r}.pkl'
    pickle_steps(retrain_path, np.array(t_retraining_steps))

    log_artifact(unlearn_path, 'unlearnSteps')
    log_artifact(retrain_path, 'retrainSteps')
    print(t_forgetting)
    print(t_retraining)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run UnlearnSPN experiments.')
    parser.add_argument('-d', type=str, required=True, help='Training dataset: msnbc, plants, adult, abalone, wine')
    parser.add_argument('-e', type=float, required=True, help='Epsilon value for Q-k-Means')
    parser.add_argument('-s', type=int, required=True, help='Random seed')
    parser.add_argument('-r', type=int, required=True, help='Number of data points to remove')
    args = parser.parse_args()
    main()
