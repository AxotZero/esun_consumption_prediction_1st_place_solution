import os
from pdb import set_trace as bp
from functools import reduce

import numpy as np
import math
import pandas as pd

from constant import target_indices


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def divide_by_sum(df):
    return df.div(df.sum(axis=1), axis=0)


def divide_by_softmax(df):
    return df.apply(softmax, axis=1)


def get_weights_by_normal_distribution(pbs):
    pbs = np.array(pbs)
    _mean = np.mean(pbs)
    _std = np.std(pbs)
    n_std = (pbs - _mean) / _std
    pvalues = np.array([ .5 * (math.erf(n / 2 ** .5) + 1) for n in n_std])
    weights = pvalues / np.sum(pvalues)
    # weights = softmax(pvalues)
    return weights



def merge_logits(df_paths, weights=None, normalize_fn=divide_by_softmax):
    if weights is None:
        weights = [1] * len(df_paths)
    assert len(weights) == len(df_paths)
    
    # get dfs
    dfs = []
    for i, path in enumerate(df_paths):
        # sort chid and set index
        df = pd.read_csv(path).sort_values(by='chid').set_index('chid')
        # normalize and apply weight
        df = normalize_fn(df) * weights[i]
        dfs.append(df)

    # sum weighted logits
    df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)

    # get top3
    topk_df_indices = df.values.argsort(axis=1)[:, -3:]
    topk_df_indices = np.flip(topk_df_indices, axis=1)
    topk_values = target_indices[topk_df_indices].astype(int)+1

    # create dataframe
    df_top3 = pd.DataFrame(
        data=np.concatenate([np.array(df.index).reshape(-1, 1), topk_values], axis=1),
        columns=['chid', 'top1', 'top2', 'top3']).astype(int).astype(str)
    df_logits = df.reset_index()
    df_logits['chid'] = df_logits['chid'].astype(int)
    return df_top3, df_logits


if __name__ == "__main__":
    ensemble_dir = '../save_dir/ensemble_result'
    ## for kfold ensemble
    dir_name = 'mm_nnbn_h192'
    kfold_dir = f'../save_dir/{dir_name}/fine'
    df_paths = [f"{kfold_dir}/fold{i}/outputs_logits.csv" for i in range(5)]
    weights = None
    save_dir = f'{ensemble_dir}/{dir_name}'

    ## for different model ensembles 
    # df_paths = [
    #     f'{ensemble_dir}/mm_cnn_hidden256_5fold/logits.csv',
    #     f'{ensemble_dir}/mm_hidden256_5fold_test/logits.csv',
    #     f'{ensemble_dir}/nn3_attn_5fold/logits.csv',
    #     f'{ensemble_dir}/mm_CnnAggBn_hidden256_5fold/logits.csv'
    # ]
    # weights = [0.8, 0.1, 0.06, 0.04]
    # # weights = get_weights_by_normal_distribution([0.72700, 0.72782, 0.72652, 0.72669])
    # dir_name = 'cnn0.8_nn0.1_nn0.06.attn_CnnAggBn0.04'
    # save_dir = f'../save_dir/ensemble_result/{dir_name}'

    for path in df_paths:
        if not os.path.exists(path):
            print(f'not exist {path}')
            exit()


    # create dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # save path and validate path
    top3_path = f"{save_dir}/top3.csv"
    logits_path = f"{save_dir}/logits.csv"
    assert not os.path.exists(top3_path)
    assert not os.path.exists(logits_path)

    # merge
    df_top3, df_logits = merge_logits(df_paths, weights=weights)
    
    
    df_top3.to_csv(top3_path, index=None)
    df_logits.to_csv(logits_path, index=None)


