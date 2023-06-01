import random

from definitions import ROOT_DIR
import pandas as pd
import os
from src.data.convert import df_to_ocel
from ocpa.objects.log.exporter.ocel import factory as ocel_export_factory
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory

OTS = [
    # TODO: Add the object types that you want to keep
]

LEAD_OT = 'LeadObject'  # TODO: Add the lead object type

START_ACTIVITIES = [
    # TODO: Add the start activities that you want to keep
]

END_ACTIVITIES = [
    # TODO: Add the end activities that you want to keep
]

NUM_EVENTS = 600000

DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'dataset.pickle')
OCEL_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'dataset.jsonocel')


def extract_from_pickle():
    print('Using pandas version', pd.__version__)

    # Read the raw extracted dataframe
    df = pd.read_pickle(DATASET_PATH)

    # only keep rows with objects
    # TODO: Add the object types that you want to keep
    df = df[(df[LEAD_OT].map(len) > 0)]

    # rename the columns
    df = df.rename(columns={
        'ID': 'event_id',
        'Type': 'event_activity',
        'Time': 'event_timestamp',
    })

    # Convert
    print('Converting to OCEL')
    ocel = df_to_ocel(df,
                      object_columns=OTS,
                      value_columns=[], )

    # Save the result as a jsonocel file
    print('Saving to jsonocel')
    ocel_export_factory.apply(ocel, OCEL_PATH)


def filter_and_sample():
    # Import the extracted jsonocel file
    print('Importing jsonocel')
    ocel = ocel_import_factory.apply(OCEL_PATH)

    # Filter process executions without a lead object
    print('Query events with lead object')
    events_with_lead_object = ocel.log.log[ocel.log.log[LEAD_OT].map(len) > 0]['event_id']

    print('Filter process executions')
    complete_process_executions = []
    for p in ocel.process_executions:
        # Filter process executions with more than 100 events
        if len(p) > 100:
            continue
        # Check if the process execution contains an event with a sales order
        if not any(e in events_with_lead_object for e in p):
            continue
        # Check if the process execution starts with the correct activity
        first_event_id = min(p)
        first_activity = ocel.get_value(first_event_id, 'event_activity')
        if first_activity not in START_ACTIVITIES:
            continue
        # Check if the process execution ends with the correct activity
        last_event_id = max(p)
        last_activity = ocel.get_value(last_event_id, 'event_activity')
        if last_activity not in END_ACTIVITIES:
            continue
        complete_process_executions.append(p)
    print('Found', len(complete_process_executions), 'complete process executions')

    print('Extracting event ids')
    # Shuffle the process executions
    random.shuffle(complete_process_executions)

    # combine all event ids of selected process execution ids
    # until we have roughly 600k events
    eids = set()
    for p in complete_process_executions:
        eids = eids | p
        if len(eids) > NUM_EVENTS:
            break

    print('Filtering events')
    # Filter based on selected event ids
    df = ocel.log.log.loc[ocel.log.log['event_id'].isin(eids)]

    # Clear memory
    del ocel

    # Reset the index and event ids
    df = df.sort_index()
    df['event_id'] = list(range(len(df)))
    df.index = list(range(len(df)))

    print('Converting to OCEL')
    # Convert small dataframe
    ocel = df_to_ocel(df, sort=False,
                      object_columns=OTS,
                      value_columns=[], )

    # Save the result as a jsonocel file
    print('Saving to jsonocel')
    ocel_export_factory.apply(ocel, OCEL_PATH.replace('.jsonocel', '_filtered.jsonocel'))


if __name__ == '__main__':
    # extract_from_pickle()
    filter_and_sample()
