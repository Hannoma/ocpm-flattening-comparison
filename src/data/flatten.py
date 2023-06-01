import logging
import os
from copy import copy

import pandas as pd
from ocpa.objects.log.ocel import OCEL
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.objects.log.exporter.ocel import factory as ocel_export_factory

from definitions import ROOT_DIR
from src.data.convert import df_to_ocel
from src.helpers.caching import CacheManager


def flatten_ocel(ocel: OCEL, object_types: list = None, resource_column: str = 'event_resource',
                 value_columns: list = None, preserve_objects: bool = False) -> pd.DataFrame:
    """Flatten the OCEL to a single event log.

    :param ocel: OCEL to be flattened
    :type ocel: :class:`OCEL <ocpa.objects.log.ocel.OCEL>`
    :param object_types: Object types to be included in the flattened event log
    :type object_types: list
    :param resource_column: Name of the resource column
    :type resource_column: str
    :param value_columns: List of value columns
    :type value_columns: list
    :param preserve_objects: Whether to preserve the objects in the flattened event log
    :type preserve_objects: bool

    :return: Flattened OCEL as event log
    :rtype: :class:`pandas.DataFrame <pandas.DataFrame>`
    """
    if value_columns is None:
        value_columns = []
    if object_types is None:
        object_types = ocel.object_types

    events = []
    for i, process_execution_ids in enumerate(ocel.process_executions):
        for event_id in process_execution_ids:
            objects, not_empty = get_objects_for_event(ocel, event_id, object_types)
            if not_empty:
                event_dict = {
                    "event_id": event_id,
                    "event_activity": ocel.get_value(event_id, "event_activity"),
                    "event_timestamp": ocel.get_value(event_id, "event_timestamp"),
                    "process_execution_id": {"p" + str(i)},
                }
                if resource_column is not None:
                    event_dict[resource_column] = ocel.get_value(event_id, resource_column)
                for value_column in value_columns:
                    event_dict[value_column] = ocel.get_value(event_id, value_column)
                if preserve_objects:
                    event_dict.update(objects)
                events.append(event_dict)

    df = pd.DataFrame(events)
    # df = df.applymap(lambda x: set() if pd.isnull(x) else x)
    return df


def introduce_case_notion(ocel: OCEL, case_column: str, preserve_objects: bool = False) -> pd.DataFrame:
    """Introduce a case notion for the OCEL to get a flattened event log.

    :param ocel: OCEL
    :type ocel: :class:`OCEL <ocpa.objects.log.ocel.OCEL>`
    :param case_column: Name of the case column
    :type case_column: str
    :param preserve_objects: Whether to preserve the objects in the flattened event log
    :type preserve_objects: bool

    :return: Flattened OCEL as event log
    :rtype: :class:`pandas.DataFrame <pandas.DataFrame>`
    """
    # extract the event log from the OCEL
    df = ocel.log.log
    # filter event log by the new case notion
    df = df[df[case_column].apply(dict_not_empty)]
    # duplicate the events for each case
    df = df.explode(case_column).reset_index(drop=True)
    # rename the case column
    df = df.rename(columns={case_column: 'concept:name'})

    if not preserve_objects:
        ots = copy(ocel.object_types)
        ots.remove(case_column)
        # remove the objects from the event log
        df = df.drop(columns=ots)

    return df


def dict_not_empty(obj):
    return obj is not None and len(obj) > 0


def get_objects_for_event(ocel: OCEL, event_id: int, object_types: list) -> (dict, bool):
    """Get the objects for an event.

    :param ocel: OCEL
    :type ocel: :class:`OCEL <ocpa.objects.log.ocel.OCEL>`
    :param event_id: Event id
    :type event_id: int
    :param object_types: Object types to be included in the flattened event log
    :type object_types: list

    :return: Objects for the event
    :rtype: dict
    :return: True if the event has objects, False otherwise
    :rtype: bool
    """
    objects = {}
    for object_type in object_types:
        objects[object_type] = ocel.get_value(event_id, object_type)

    not_empty = any(map(dict_not_empty, objects.values()))

    return objects, not_empty


def apply(ocel_path: str, dataset: dict, object_types: list = None) -> str:
    """Flatten the OCEL to a single event log.

    :param ocel_path: Path to the OCEL
    :type ocel_path: str
    :param dataset: Dataset
    :type dataset: dict
    :param object_types: Object types to be included in the flattened event log
    :type object_types: list

    :return: Flattened OCEL as event log
    :rtype: :class:`pandas.DataFrame <pandas.DataFrame>`
    """
    logger = logging.getLogger(__name__)

    # initialize cache
    cache = CacheManager(os.path.join(ROOT_DIR, 'cache', 'flattening'))

    # check if features are already cached
    cache_key = {
        'ocel_path': ocel_path,
        'object_types': copy(object_types),
        'version': 2,
    }
    flat_ocel_path = cache.get_file_path(cache_key, 'jsonocel')
    if cache_key in cache:
        logger.info('Loading flattened log from cache')
        return flat_ocel_path
    logger.info('Flattening the log because these parameters are not cached. This may take a while')

    # load the OCEL
    ocel = ocel_import_factory.apply(ocel_path)

    if object_types is None:
        object_types = ocel.object_types

    # flatten the OCEL
    if not len(object_types) == 1:
        logger.info(f'Flattening the log based on {object_types} object types')
        flat_log = flatten_ocel(ocel, object_types, dataset['columns']['resource'], dataset['columns']['value_columns'])
        flattened_object_types = ['process_execution_id']
    else:
        logger.info(f'Introducing a case notion based on {object_types[0]} object type')
        flat_log = introduce_case_notion(ocel, object_types[0])
        flattened_object_types = ['concept:name']

    # convert the flattened log to an OCEL
    ocel = df_to_ocel(flat_log, object_columns=flattened_object_types, sort=False)

    # save the flattened log
    logger.info('Saving flattened log to cache')
    cache[cache_key] = cache_key

    # write the flattened log to a file
    ocel_export_factory.apply(ocel, flat_ocel_path)

    return flat_ocel_path
