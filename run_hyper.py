import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

from run_single import run_single


class HyperParamLoader():
    def __init__(self, arg_range):
        self.arg_range = arg_range
        self.k_list = list(arg_range.keys())
        self.cur_layer = 0
        self.choice = np.zeros((len(self.k_list), 1))
        self.args = []

        self._dfs(0)
        assert len(self.args) == np.prod([len(_) for _ in arg_range.values()])
        self.n_args = len(self.args)
        print('n_args', self.n_args, flush=True)

    def _dfs(self, layer):
        if layer == len(self.k_list):
            ans = {}
            for l, k in enumerate(self.k_list):
                ans[k] = self.arg_range[k][int(self.choice[l])]
            self.args.append(ans)
            return

        k = self.k_list[layer]
        rg = self.arg_range[k]
        for i in range(len(rg)):
            self.choice[layer] = i
            self._dfs(layer + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='SASRec', help='model name')
    parser.add_argument('-d', type=str, default='Sports', help='dataset name')
    parser.add_argument('-e', type=str, default='test', help='name of experiments')
    parser.add_argument('-hp', type=str, default='hyper_props/norm.json', help='hyper props path')
    args, unparsed = parser.parse_known_args()
    print(args)

    # Load hyper-parameter logs
    with open(args.hp, 'r') as f:
        arg_range = json.load(f)


    os.makedirs('hyper_res', exist_ok=True)

    # Init hyper tuning logs
    LOG_NAME = os.path.join('hyper_res', '-'.join(sys.argv).replace('/', '|'))
    hlog = open(LOG_NAME, 'a+')
    hlog.write('======= META =======\n')
    hlog.write(str(args) + '\n\n')
    hlog.write(str(arg_range) + '\n\n')
    hlog.flush()

    # Start hyper tuning
    hp_loader = HyperParamLoader(arg_range)
    best_val_score = best_valid = best_test = best_round = best_params = best_model_path = None
    base_func = run_single

    for j, hyper_params in enumerate(tqdm(hp_loader.args)):
        kwargs = {
            'model_name': args.m,
            'dataset_name': args.d
        }
        kwargs.update(hyper_params)
        model, dataset, ret = base_func(**kwargs)

        hlog.write(f'======= Round {j} =======\n')
        hlog.write(str(hyper_params) + '\n\n')
        hlog.write('Best Valid Result: ' + str(ret['best_valid_result']) + '\n\n')
        hlog.flush()

        if best_val_score is None or ret['best_valid_score'] > best_val_score:
            hlog.write(f'\n Best Valid Updated: {best_val_score} -> {ret["best_valid_score"]}\n\n')
            hlog.flush()

            best_val_score = ret['best_valid_score']
            best_valid = ret['best_valid_result']
            best_test = ret['test_result']
            best_round = j
            best_params = hyper_params
            best_model_path = ret['model_path']

    hlog.write('======= FINAL =======\n')
    hlog.write(f'Best Round: {best_round}\n')
    hlog.write(f'Best Valid Score: {best_val_score}\n')
    hlog.write('Best Params: ' + str(best_params) + '\n')
    hlog.write('Final Valid Result: ' + str(best_valid) + '\n')
    hlog.write('Final Test Result: ' + str(best_test) + '\n')
    hlog.close()

    os.system(f'mv {best_model_path} checkpoints/{args.m}-{args.d}-{args.e}.pth')
