import pandas as pd
from tqdm import tqdm

from ocpa.algo.predictive_monitoring.obj import Feature_Storage


def tabular_encoding(feature_storage: Feature_Storage, target: tuple):
    # Convert the feature storage to a tabular feature storage
    train_table = construct_table(feature_storage, index_list=feature_storage.training_indices, k=8)
    test_table = construct_table(feature_storage, index_list=feature_storage.test_indices, k=8)
    # Split the data into features and target
    y_train, y_test = train_table[target], test_table[target]
    X_train, X_test = train_table.drop(target, axis=1), test_table.drop(target, axis=1)
    return X_train, y_train, X_test, y_test


def construct_table(feature_storage, index_list="all", k=0):
    if index_list == "all":
        fgs = feature_storage.feature_graphs
    else:
        fgs = list(filter(lambda x: x.pexec_id in index_list, feature_storage.feature_graphs))

    fgs.sort(key=lambda x: x.pexec_id)

    dict_list = []
    for g in tqdm(fgs):
        nodes = g.nodes
        nodes.sort(key=lambda x: x.event_id)
        for i, node in enumerate(g.nodes):
            if i >= k - 1:
                dict_list.append(node.attributes)
    df = pd.DataFrame(dict_list)
    return df
