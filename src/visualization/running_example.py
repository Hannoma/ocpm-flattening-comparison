import datetime

import networkx as nx
import pandas as pd
from faker import Faker
from ocpa.algo.predictive_monitoring import tabular
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring

from ocpa.algo.predictive_monitoring import sequential
from src.data.convert import df_to_ocel
from src.data.flatten import flatten_ocel, introduce_case_notion
from src.visualization.visualizations import convert_feature_graph_to_networkx_graph, to_tikz_sequence, to_latex_table, \
    to_tikz_figure

ots = ['Order', 'Item']

log = [
    # first process execution
    {'activity': 'create order', 'Order': [1], 'Item': [1, 2]},
    {'activity': 'approve order', 'Order': [1]},
    {'activity': 'approve item', 'Item': [1]},
    {'activity': 'approve item', 'Item': [2]},
    {'activity': 'create invoice', 'Order': [1]},
    {'activity': 'send confirmation', 'Order': [1]},
    {'activity': 'create delivery', 'Order': [1], 'Item': [1, 2]},
    # second process execution
    {'activity': 'create order', 'Order': [2], 'Item': [3]},
    {'activity': 'approve item', 'Item': [3]},
    {'activity': 'approve order', 'Order': [2]},
    {'activity': 'add item', 'Order': [2], 'Item': [4]},
    {'activity': 'send confirmation', 'Order': [2]},
    {'activity': 'create invoice', 'Order': [2]},
    {'activity': 'create delivery', 'Order': [2], 'Item': [3, 4]},
]

# create dataframe
Faker.seed(42)
faker = Faker()
df = pd.DataFrame(log)
for ot in ots:
    df[ot] = df[ot].apply(lambda x: set([ot[0].lower() + str(i) for i in x]) if isinstance(x, list) else set())
df['event id'] = df.index
start_date = datetime.datetime(2021, 7, 3)
df['timestamp'] = df['event id'].apply(lambda x:
                                       faker.date_time_between(start_date=start_date + datetime.timedelta(days=x),
                                                               end_date=start_date + datetime.timedelta(days=x + 1)))
df = df[['event id', 'activity'] + ots + ['timestamp']]

to_latex_table(df, 'running_example.tex', caption='Running example', label='tab:running_example')

# Rename the event attributes to have the event prefix
df = df.rename(columns={'event id': 'event_id', 'activity': 'event_activity', 'timestamp': 'event_timestamp'})
for ot in ots:
    df[ot] = df[ot].apply(lambda x: str(list(x)))
# Convert the dataframe to an ocel
ocel = df_to_ocel(df, object_columns=ots)

# Flatten the ocel based on a compound object type
df_compound = flatten_ocel(ocel, resource_column=None, preserve_objects=True)


# Merge the object columns
def join(x):
    x = list(x)
    # Check if the list contains strings
    if all(isinstance(e, str) for e in x):
        # Sort the list
        x.sort()
    return ' AND '.join(x)


df_compound[join(ots)] = df_compound[ots].apply(lambda x: join([join(y) for y in x if len(y) > 0]), axis=1)
df_compound = df_compound.drop(columns=ots)
# map the process execution to a string
df_compound['process execution id'] = df_compound['process_execution_id'].apply(lambda x: ''.join(x))
df_compound = df_compound.drop(columns=['process_execution_id'])
# Reverse the renaming of the event attributes
df_compound = df_compound.rename(
    columns={'event_id': 'event id', 'event_activity': 'activity', 'event_timestamp': 'timestamp'})
to_latex_table(df_compound, 'flattening_compound-case-notion.tex', label='tab:flattening_compound-case-notion',
               caption='Flattened the log based on a compound object type')

# Flatten the ocel based on Item
df_single = introduce_case_notion(ocel, case_column='Item')
# Reverse the renaming of the event attributes
df_single = df_single.rename(columns={'event_id': 'event id', 'event_activity': 'activity',
                                      'event_timestamp': 'timestamp', 'concept:name': 'Item'})
# Sort by event id and item
df_single = df_single.sort_values(by=['event id', 'Item'])
to_latex_table(df_single, 'flattening_single-case-notion.tex', caption='Flattened the log based on object type Item',
               label='tab:flattening_single-case-notion')

# Visualize the process executions
nx_variant_graphs = []
for i, variant in enumerate(ocel.variant_graphs):
    variant_graph = ocel.variant_graphs[variant][0]
    nx.set_node_attributes(variant_graph, {
        n: ocel.get_value(n, 'event_activity').replace(' ', r'\\')
        for n in variant_graph.nodes
    }, name='node_label')
    nx_variant_graphs.append(variant_graph)

to_tikz_figure(nx_variant_graphs, 'process-executions.tex', caption='Process executions',
               label='fig:process-executions', scale=0.0125)

# Build the feature storage
FEATURES = {
    'RT': (predictive_monitoring.EVENT_REMAINING_TIME, ()),
    'ET': (predictive_monitoring.EVENT_ELAPSED_TIME, ()),
    'PAC': (predictive_monitoring.EVENT_PREVIOUS_ACTIVITY_COUNT, ('approve item',)),
}
feature_storage = predictive_monitoring.apply(ocel, list(FEATURES.values()), [])

# Visualize the tabular encoding
tabular_encoding_df = tabular.construct_table(feature_storage)
tabular_encoding_df = tabular_encoding_df.rename(columns={f: k for k, f in FEATURES.items()})
to_latex_table(tabular_encoding_df, 'encoding_tabular.tex', caption='Tabular encoding',
               label='tab:encoding_tabular')


def convert_features_to_label(features) -> str:
    return r'\\'.join([rf'\textbf{{{k}:}} {features[f]}' for k, f in FEATURES.items()])


# Visualize the sequence encoding
sequences = sequential.construct_sequence(feature_storage)
sequences = list(map(lambda seq: list(map(convert_features_to_label, seq)), sequences))
to_tikz_sequence(sequences, 'encoding_sequence.tex', caption='Sequence encoding',
                 label='fig:encoding_sequence')

# Visualize the feature graphs
nx_fgs = []
for i, feature_graph in enumerate(feature_storage.feature_graphs):
    nx_fg = convert_feature_graph_to_networkx_graph(feature_graph)
    nx.set_node_attributes(nx_fg, {
        n[0]: convert_features_to_label(n[1]['features'])
        for n in nx_fg.nodes(data=True)
    }, name='node_label')
    nx_fgs.append(nx_fg)

to_tikz_figure(nx_fgs, f'feature-graphs.tex', caption=f'Feature graphs',
               label=f'fig:feature-graphs', scale=0.02)
