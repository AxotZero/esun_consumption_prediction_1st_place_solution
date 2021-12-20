import os
from pdb import set_trace as bp
from functools import reduce

import numpy as np
import pandas as pd

from constant import target_indices


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def divide_by_sum(df):
    return df.div(df.sum(axis=1), axis=0)


def divide_by_softmax(df):
    return df.apply(softmax, axis=1)


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

    # bp()
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
    # save_5fold_dir = '../save_dir/mm_hidden256_5fold_test/fine'
    save_5fold_dir = '../save_dir/mm_cnn_hidden256_5fold/fine'
    # df_paths = [f"{save_5fold_dir}/fold{i}/outputs_logits.csv" for i in range(5)]
    df_paths = [
        '../save_dir/mm_cnn_hidden256_5fold/fine/kfold_logits.csv',
        '../cc/save_dir/mm_hidden256_5fold_test/fine/kfold_top3_2.csv'
    ]

    # merge
    df_top3, df_logits = merge_logits(df_paths, weights=[0.7, 0.3])
    
    # save
    df_top3.to_csv(f"{save_5fold_dir}/en_mlp_kfold_top3.csv", index=None)
    df_logits.to_csv(f"{save_5fold_dir}/en_mlp_kfold_logits.csv", index=None)


