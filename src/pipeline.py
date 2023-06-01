import logging
from pprint import pprint

import pandas as pd
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring

from definitions import ROOT_DIR
from src.data import flatten
from src.data.convert import csv_to_jsonocel
from src.features.build_features import build_features, ALL_ACTIVITIES, ALL_RESOURCES, ALL_OBJECT_TYPES, \
    construct_lookup_table, add_custom_features, LOOKUP_FEATURE, ALL_NOMINAL_ATTRIBUTES, ALL_NUMERIC_ATTRIBUTES, \
    OHE_FEATURE, NUMERIC_FEATURE
from src.helpers.caching import CacheManager
from src.helpers.dataset import get_selected_dataset, log_path_for_dataset
from src.models.gnn import train_model_with_gnn
from src.models.lstm import train_model_with_lstm
from src.models.train_model import train_model_with_feature_storage, train_model_with_graph_embedding, \
    train_baseline_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

TARGET_FEATURE = (predictive_monitoring.EVENT_REMAINING_TIME, ())
BASELINE_FEATURE = (predictive_monitoring.EVENT_IDENTITY, ())
BASIC_FEATURES = [
    # C5
    (predictive_monitoring.EVENT_ACTIVITY, ALL_ACTIVITIES),

    # D3
    # (NUMERIC_FEATURE, (ALL_NUMERIC_ATTRIBUTES, 'MEAN')),
    # (OHE_FEATURE, ALL_NOMINAL_ATTRIBUTES),

    # R3
    # (predictive_monitoring.EVENT_RESOURCE, ('event_resource', ALL_RESOURCES)), too many resources?
    # (predictive_monitoring.EVENT_CURRENT_TOTAL_WORKLOAD, ()),     # do not use for now

    # P9
    (predictive_monitoring.EVENT_SERVICE_TIME, ('event_start_timestamp', )),
]
ENHANCED_FEATURES = [
    # C1, C2, C3
    (predictive_monitoring.EVENT_CURRENT_ACTIVITIES, ALL_ACTIVITIES),
    (predictive_monitoring.EVENT_PRECEDING_ACTIVITES, ALL_ACTIVITIES),
    (predictive_monitoring.EVENT_PREVIOUS_ACTIVITY_COUNT, ALL_ACTIVITIES),
    # C4 does not exist yet

    # D1, D2 skipped for now because no meaningful aggregation functions are available
    # (predictive_monitoring.EVENT_AGG_PREVIOUS_CHAR_VALUES, ()),
    # (predictive_monitoring.EVENT_PRECEDING_CHAR_VALUES, ()),

    # R1
    # (predictive_monitoring.EVENT_CURRENT_RESOURCE_WORKLOAD, ()),  # do not use for now

    # P1, P2, P6, P10 (P3 is used as target feature)
    # (predictive_monitoring.EVENT_EXECUTION_DURATION, ()), # no extraction function available
    (predictive_monitoring.EVENT_ELAPSED_TIME, ()),
    (predictive_monitoring.EVENT_SOJOURN_TIME, ()),
    (predictive_monitoring.EVENT_WAITING_TIME, ('event_start_timestamp',)),
]
OCEL_FEATURES = [
    # P4, P5, P7, P8
    (predictive_monitoring.EVENT_FLOW_TIME, ()),
    (predictive_monitoring.EVENT_SYNCHRONIZATION_TIME, ()),
    (predictive_monitoring.EVENT_POOLING_TIME, ALL_OBJECT_TYPES),
    # (predictive_monitoring.EVENT_LAGGING_TIME, ALL_OBJECT_TYPES),     # TODO this feature has bugs

    # O1, O2, O3, O4, O5, O6
    # (predictive_monitoring.EVENT_CURRENT_TOTAL_OBJECT_COUNT, ()),   # how to set time window?
    (predictive_monitoring.EVENT_PREVIOUS_OBJECT_COUNT, ()),
    (predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT, ALL_OBJECT_TYPES),
    # (predictive_monitoring.EVENT_OBJECTS, ALL_OBJECTS),         # too many objects
    (predictive_monitoring.EVENT_NUM_OF_OBJECTS, ()),
    (predictive_monitoring.EVENT_TYPE_COUNT, ALL_OBJECT_TYPES),
]


def generate_combinations(c: dict):
    import itertools
    keys, values = zip(*c.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def train_and_score_model(d: dict, c: dict, target) -> tuple[object, dict]:
    dataset_path = log_path_for_dataset(d, 'jsonocel')

    # flatten the event log
    if c['flattening'] == 'ALL':
        dataset_path = flatten.apply(dataset_path, d)
    elif type(c['flattening']) is list:
        dataset_path = flatten.apply(dataset_path, d, object_types=c['flattening'])

    # build features
    feature_storage = build_features(dataset_path, feature_set=c['feature_set'])

    # Split the data into training and test set
    logging.info('Normalizing and splitting data into training and test set')
    feature_storage.extract_normalized_train_test_split(test_size=0.2, state=42)

    logging.info('Encoding features and training model')
    if c['model'] == 'baseline':
        return train_baseline_model(feature_storage, target, c['encoding'])

    if c['encoding'] == 'tabular':
        return train_model_with_feature_storage(feature_storage=feature_storage,
                                                target=target,
                                                model_type=c['model'])
    elif c['encoding'] == 'graph_embedding':
        return train_model_with_graph_embedding(feature_storage=feature_storage,
                                                target=target,
                                                model_type=c['model'])
    elif c['encoding'] == 'sequence':
        return train_model_with_lstm(feature_storage=feature_storage,
                                     target=target,
                                     dataset_name=d['filename'])
    elif c['encoding'] == 'graph':
        return train_model_with_gnn(feature_storage=feature_storage,
                                    target=target,
                                    dataset_name=d['filename'])
    else:
        raise ValueError(f'Unknown encoding {c["encoding"]}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    dataset = get_selected_dataset()

    if dataset['format'] == 'csv':
        logging.info(f'Converting {dataset["filename"]} to jsonocel')
        csv_to_jsonocel(dataset)
        logging.info(f'Finished converting {dataset["filename"]} to jsonocel')

    # add lookup feature to the feature extractor
    add_custom_features()
    # extract target feature from original event log
    target_storage = build_features(log_path_for_dataset(dataset, 'jsonocel'), feature_set=[TARGET_FEATURE])
    # construct lookup feature to look up original feature
    target_lookup_table = construct_lookup_table(target_storage, TARGET_FEATURE)
    lookup_feature = (LOOKUP_FEATURE, (target_lookup_table, ))

    configuration = {
        'flattening': [None, 'ALL'],
        'feature_set': [
            [lookup_feature, BASELINE_FEATURE],
            [lookup_feature] + BASIC_FEATURES,
            [lookup_feature] + BASIC_FEATURES + ENHANCED_FEATURES,
            [lookup_feature] + BASIC_FEATURES + ENHANCED_FEATURES + OCEL_FEATURES,
        ],
        'encoding': ['graph_embedding'],
        'model': [['random_forest', 'linear_regression', 'regression_tree', 'mlp']],
    }

    results = CacheManager(os.path.join(ROOT_DIR, 'cache', 'results'))
    combinations = generate_combinations(configuration)

    # Overwrite existing results
    force = True

    # Run pipeline
    for com in combinations:
        results_key = {
            'dataset': dataset,
            'configuration': com
        }

        # results.repair(results_key)

        if results_key in results:
            if not force:
                print('Already trained a model for:')
                pprint(com)
                continue
            else:
                results.delete(results_key)

        print('Training a model for:')
        pprint(com)
        model, scores = train_and_score_model(dataset, com, lookup_feature)

        result = {
            'model': model,
            'scores': scores
        }
        results.save(results_key, result)

    # Combine results
    results_df = pd.DataFrame()
    for com in combinations:
        results_key = {
            'dataset': dataset,
            'configuration': com
        }

        result = results.load(results_key)

        if isinstance(result['scores'], list):
            result['scores'] = dict(zip(['mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'mape'], result['scores']))
        if type(com['model']) is list:
            for model in com['model']:
                results_df = results_df.append({
                    'flattening': com['flattening'],
                    'features': len(com['feature_set']),
                    'encoding': com['encoding'],
                    'model': model,
                    **result['scores'][model]}, ignore_index=True)
        else:
            results_df = results_df.append({
                'flattening': com['flattening'],
                'features': len(com['feature_set']),
                'encoding': com['encoding'],
                'model': com['model'],
                **result['scores']}, ignore_index=True)
    results_df.to_csv(os.path.join(ROOT_DIR, 'models', dataset['filename'], 'results.csv'), index=False)


