import logging
import os

import pandas as pd

from ocpa.objects.log.ocel import OCEL

from definitions import ROOT_DIR
from src.helpers.caching import CacheManager

from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory

from src.helpers.dataset import get_selected_dataset
from src.helpers.lookup_table import LookupTable

ALL_ACTIVITIES = 'ALL ACTIVITIES'
ALL_RESOURCES = 'ALL RESOURCES'
ALL_OBJECTS = 'ALL OBJECTS'
ALL_OBJECT_TYPES = 'ALL OBJECT TYPES'
ALL_NUMERIC_ATTRIBUTES = 'ALL NUMERIC ATTRIBUTES'
ALL_NOMINAL_ATTRIBUTES = 'ALL NOMINAL ATTRIBUTES'
LOOKUP_FEATURE = 'lookup_feature'
NUMERIC_FEATURE = 'numeric_feature'
OHE_FEATURE = 'ohe_feature'


def build_features(ocel_path: str, feature_set: list[tuple[str, any]]) -> Feature_Storage:
    logger = logging.getLogger(__name__)

    # initialize cache
    cache = CacheManager(os.path.join(ROOT_DIR, 'cache', 'features'))

    # check if features are already cached
    cache_key = {
        'ocel_path': ocel_path,
        'feature_set': feature_set,
    }
    if cache_key in cache:
        logger.info('Loading features from cache')
        return cache.load(cache_key)
    logger.info('Building features because these parameters are not cached. This may take a while')

    # load log
    ocel = ocel_import_factory.apply(ocel_path)

    # transform features
    new_feature_set = transform_features(ocel, feature_set)

    # extract features
    feature_storage = predictive_monitoring.apply(ocel, event_based_features=new_feature_set, workers=8)

    # save features to cache
    cache.save(cache_key, feature_storage)
    logger.info('Saved features to cache')
    return feature_storage


def transform_features(ocel: OCEL, feature_set: list[tuple[str, any]]) -> list[tuple[str, any]]:
    """
    Transforms the feature set to a format that is compatible with the feature extraction algorithm.

    :param ocel: The log that should be used to extract the features.
    :type ocel: OCEL
    :param feature_set: The feature set that should be transformed.
    :type feature_set: list[tuple[str, any]]

    :return: The transformed feature set.
    :rtype: list[tuple[str, any]]
    """
    dataset = get_selected_dataset()

    all_activities = list(ocel.obj.activities)
    all_objects = ocel.obj.raw.obj_ids
    all_object_types = ocel.object_types

    resource_column = dataset['columns']['resource']
    if resource_column is not None:
        all_resources = list(set(ocel.log.log[resource_column].tolist()))
    else:
        all_resources = []

    if 'value_columns' in dataset['columns']:
        numeric_columns = [k for k, v in dataset['columns']['value_columns'].items() if v == 'numeric']
        nominal_columns = [k for k, v in dataset['columns']['value_columns'].items() if v == 'nominal']
    else:
        numeric_columns = []
        nominal_columns = []

    new_feature_set = []
    for feature_name, feature_params in feature_set:
        if feature_params == ALL_ACTIVITIES:
            new_feature_set.extend([(feature_name, (activity,)) for activity in all_activities])
        elif len(feature_params) > 1 and feature_params[1] == ALL_ACTIVITIES:
            new_feature_set.extend([(feature_name, (feature_params[0], activity)) for activity in all_activities])

        elif feature_params == ALL_RESOURCES:
            new_feature_set.extend([(feature_name, (resource,)) for resource in all_resources])
        elif len(feature_params) > 1 and feature_params[1] == ALL_RESOURCES:
            new_feature_set.extend([(feature_name, (feature_params[0], resource)) for resource in all_resources])

        elif feature_params == ALL_OBJECTS:
            new_feature_set.extend([(feature_name, (obj,)) for obj in all_objects])
        elif feature_params == ALL_OBJECT_TYPES:
            new_feature_set.extend([(feature_name, (object_type,)) for object_type in all_object_types])
        elif len(feature_params) > 1 and feature_params[0] == ALL_NUMERIC_ATTRIBUTES:
            orig_value = feature_params[1]
            for numeric_attribute in numeric_columns:
                if orig_value == 'MEAN':
                    value = ocel.log.log[numeric_attribute].mean()
                else:
                    value = orig_value
                new_feature_set.append((feature_name, (numeric_attribute, value)))
        elif feature_params == ALL_NOMINAL_ATTRIBUTES:
            for nominal_attribute in nominal_columns:
                unique_values = ocel.log.log[nominal_attribute].unique()
                for value in unique_values:
                    # Fill missing values with `None` to avoid errors in the feature extraction algorithm
                    if pd.isna(value):
                        value = None
                    new_feature_set.append((feature_name, (nominal_attribute, value)))
        else:
            new_feature_set.append((feature_name, feature_params))
    return new_feature_set


def lookup_feature(node, ocel: OCEL, params: tuple[dict[str, any]]):
    """
    This feature returns the value for the given event id from the lookup table.
    """
    e_id = node.event_id
    lookup_table = params[0]
    return lookup_table[str(e_id)]


def ohe_feature(node, ocel: OCEL, params: tuple[str, str]):
    """
    This feature returns the one-hot-encoded value for the given event id from the lookup table.
    """
    nominal_column, value = params
    event_value = ocel.get_value(node.event_id, nominal_column)
    if event_value == value or pd.isna(event_value) and value is None:
        return 1
    else:
        return 0


def numeric_feature(node, ocel: OCEL, params: tuple[str, int]):
    """
    This feature returns the one-hot-encoded value for the given event id from the lookup table.
    """
    numeric_column, default_value = params
    value = ocel.get_value(node.event_id, numeric_column)
    if value is None or pd.isna(value):
        return default_value
    else:
        return value


def add_custom_features():
    """
    This function adds new features to the feature extraction algorithm
    """
    predictive_monitoring.VERSIONS[predictive_monitoring.EVENT_BASED][LOOKUP_FEATURE] = lookup_feature
    predictive_monitoring.VERSIONS[predictive_monitoring.EVENT_BASED][OHE_FEATURE] = ohe_feature
    predictive_monitoring.VERSIONS[predictive_monitoring.EVENT_BASED][NUMERIC_FEATURE] = numeric_feature


def construct_lookup_table(feature_storage: Feature_Storage, target) -> LookupTable:
    table = {}
    for g in feature_storage.feature_graphs:
        for node in g.nodes:
            table[str(node.event_id)] = node.attributes[target]
    return LookupTable(table)
