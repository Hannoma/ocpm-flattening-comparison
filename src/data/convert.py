import pandas as pd
from ast import literal_eval
from ocpa.objects.log.ocel import OCEL
from ocpa.objects.log.variants.table import Table
from ocpa.objects.log.variants.graph import EventGraph
import ocpa.objects.log.converter.versions.df_to_ocel as obj_converter
import ocpa.objects.log.variants.util.table as table_utils
from ocpa.objects.log.exporter.ocel import factory as ocel_export_factory

from src.helpers.dataset import log_path_for_dataset


def df_to_ocel(df: pd.DataFrame, activity_column: str = 'event_activity', timestamp_column: str = 'event_timestamp',
               object_columns: list = None, value_columns: list = None, event_id_column: str = 'event_id', sort=True) -> OCEL:
    """Convert a dataframe to an OCEL.

    :param df: Dataframe
    :type df: :class:`pandas.DataFrame <pandas.DataFrame>`
    :param activity_column: Name of the activity column
    :type activity_column: str
    :param timestamp_column: Name of the timestamp column
    :type timestamp_column: str
    :param object_columns: Names of object columns
    :type object_columns: list
    :param value_columns: Names of value columns
    :type value_columns: list
    :param event_id_column: Name of the event id column
    :type event_id_column: str
    :param sort: Whether to sort the dataframe by timestamp
    :type sort: bool

    :return: OCEL
    :rtype: :class:`OCEL <ocpa.objects.log.ocel.OCEL>`
    """
    if object_columns is None:
        object_columns = []
    if value_columns is None:
        value_columns = []

    # Convert object columns to sets
    def _convert_to_set(x):
        try:
            if isinstance(x, set):
                return x
            elif isinstance(x, list):
                return set(x)
            elif isinstance(x, str) and not x.startswith('{'):
                return {x}
            else:
                return literal_eval(x.replace('set()', '{}'))
        except Exception as e:
            print(e)
            return []
    for col in object_columns:
        df[col] = df[col].apply(_convert_to_set)

    if sort:
        print('Sorting dataframe...')
        # Sort the dataframe by timestamp
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(by=timestamp_column)

        # Reset the index and event ids
        df[event_id_column] = list(range(len(df)))
        df.index = list(range(len(df)))

    parameters = {
        'obj_names': object_columns,
        'val_names': value_columns,
        'act_name': activity_column,
        'time_name': timestamp_column,
    }
    log = Table(df, parameters=parameters)
    objects = obj_converter.apply(df)
    graph = EventGraph(table_utils.eog_from_log(log))
    ocel = OCEL(log, objects, graph, parameters=parameters)
    return ocel


def csv_to_jsonocel(dataset: dict):
    df = pd.read_csv(log_path_for_dataset(dataset))
    ocel = df_to_ocel(df=df,
                      object_columns=dataset['columns']['object_columns'],
                      value_columns=dataset['columns']['value_columns'].keys(),)
    # write the flattened log to a file
    ocel_export_factory.apply(ocel, log_path_for_dataset(dataset, 'jsonocel'))
