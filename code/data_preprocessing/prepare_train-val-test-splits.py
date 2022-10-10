import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def map_ids(mapping, indices):
    return np.array([mapping[i] for i in indices])


def splitter(X, y, mapping):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1234)
    split_dict = dict()
    split_id = 1

    for rem_index, test_index in rskf.split(X, y):
        X_rem, y_rem = X[rem_index], y[rem_index]
        X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=0.125, random_state=1234,
                                                          stratify=y_rem)
        train_index, val_index = np.sort(X_train.index.to_numpy()), np.sort(X_val.index.to_numpy())

        split_dict[split_id] = {'train_index': map_ids(mapping, train_index),
                                'val_index': map_ids(mapping, val_index),
                                'test_index': map_ids(mapping, test_index)}
        split_id += 1

    return split_dict


if __name__ == '__main__':
    df = pd.read_csv("../../data/dataset+labels.csv")

    # label transformation
    concr_dict = {
        'high concreteness': 2,
        'intermediate concreteness': 1,
        'low concreteness': 0
    }
    df['concreteness_num'] = df['concreteness'].apply(lambda x: concr_dict[x])

    # X, y for the joint and the two subsets
    X_all, y_all = df['text'], df['concreteness_num']

    df_mpos = df[df['code'] == "mpos"].reset_index()
    df_prem = df[df['code'] == "premise"].reset_index()

    X_mpos, y_mpos = df_mpos['text'], df_mpos['concreteness_num']
    X_prem, y_prem = df_prem['text'], df_prem['concreteness_num']

    # create mappings to reassign index of original dataframe
    mapping_all = {i: i for i in df.index}
    mapping_mpos = {i: df_mpos.iloc[i]['index'] for i in df_mpos.index}
    mapping_prem = {i: df_prem.iloc[i]['index'] for i in df_prem.index}

    # compute splits: 10 times 5CV 70%-10%-20%
    splits_all = splitter(X_all, y_all, mapping_all)
    splits_mpos = splitter(X_mpos, y_mpos, mapping_mpos)
    splits_prem = splitter(X_prem, y_prem, mapping_prem)

    # pickle splits
    a = {'joint': splits_all, 'mpos': splits_mpos, 'premise': splits_prem}
    with open('../../data/splits.pickle', 'wb') as handle:
        pickle.dump(a, handle)
