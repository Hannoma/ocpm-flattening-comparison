import os
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from definitions import  ROOT_DIR
from src.helpers.dataset import get_selected_dataset

FIGURE_WRAPPER = r"""\begin{{figure}}
    {content}
    {caption}
    {label}
\end{{figure}}
"""


def convert_feature_graph_to_networkx_graph(feature_graph: Feature_Storage.Feature_Graph) -> nx.DiGraph:
    """
    Convert a feature graph to a networkx graph

    :param feature_graph: The feature graph to convert
    :type feature_graph: Feature_Storage.Feature_Graph

    :return: The converted networkx graph
    :rtype: nx.DiGraph
    """
    graph = nx.DiGraph(incoming_graph_data=None, **feature_graph.attributes)
    graph.add_nodes_from(
        (node.event_id, {'features': node.attributes}) for node in feature_graph.nodes)
    graph.add_edges_from((edge.source, edge.target, {'features': edge.attributes}) for edge in
                         feature_graph.edges)
    return graph


def to_latex_table(df, filename, caption=None, label=None):
    style = df.style.hide_index() \
        .format_index("\\textbf{{{}}}", escape="latex", axis=1) \
        .format(lambda x: str(x).replace('set()', '{}').replace("{", '\\{').replace("}", '\\}').replace("'", ''))

    col_format = 'l' + 'c' * (len(df.columns) - 1)
    output = style.to_latex(hrules=False, caption=caption, label=label, column_format=col_format)
    output = output.replace(r'\begin{tabular}{' + col_format + '}', r"""
    \centering
    \begin{NiceTabular}{""" + col_format + r"""}
    \CodeBefore
    \rowcolor{gray!50}{1}
    \rowcolors{2}{gray!25}{white}
    \Body""")
    output = output.replace(r'\end{tabular}', r'\end{NiceTabular}')

    if filename:
        with open(os.path.join(ROOT_DIR, 'reports', 'visualizations', filename), 'w') as f:
            f.write(output)

    return output


def to_tikz_figure(gs: list[nx.DiGraph], filename, caption=None, label=None, scale: float = 1, positions=None):
    if positions is None:
        positions = []
        for g in gs:
            pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
            # Scale the pos to fit the figure
            pos = {n: (y * scale * -2, x * scale) for n, (x, y) in pos.items()}
            positions.append(pos)

    output = nx.nx_latex.to_latex(gs, pos=positions, as_document=False, caption=caption, latex_label=label,
                                  n_rows=1, figure_wrapper=FIGURE_WRAPPER,
                                  default_node_options=r'[align=center,font=\tiny]')
    if filename:
        with open(os.path.join(ROOT_DIR, 'reports', 'visualizations', filename), 'w') as f:
            f.write(output)

    return output


def to_tikz_sequence(seq: list, filename, caption=None, label=None):
    output = r"""
\begin{figure}[ht]
    \begin{tikzpicture}[nodes={shape=signal,signal from=west, signal to=east,
        align=center,fill=myc,on chain,minimum height=2.5em,font=\tiny,
        inner xsep=0.5em},start chain=going right,node distance=0.5ex]
        PATH
    \end{tikzpicture}
    \caption{CAPTION}
    \label{LABEL}
\end{figure}"""
    path = r'\path '
    nodes = []
    for i, s in enumerate(seq):
        options = ''
        if i == 0:
            options = '[signal from=nowhere]'
        else:
            options = ''
        nodes.append(f'node{options}{{{s}}}')
    path += ' '.join(nodes) + ';'
    output = output.replace('PATH', path).replace('CAPTION', caption).replace('LABEL', label)

    if filename:
        with open(os.path.join(ROOT_DIR, 'reports', 'visualizations', filename), 'w') as f:
            f.write(output)
    return output


def visualize_results():
    dataset = get_selected_dataset()

    results_df = pd.read_csv(os.path.join(ROOT_DIR, 'models', dataset['filename'], 'results_combined.csv'))

    # Add the feature names to the results
    feature_mapping = {
        2: 'Baseline',
        4: 'Basic',
        10: 'Basic + Enhanced',
        17: 'All',
    }
    results_df['features'] = results_df['features'].map(feature_mapping)
    # Clean up the flattening column
    results_df['flattening'] = results_df['flattening'].map(lambda x: 'Compound object type' if x == 'ALL' else 'None')

    # Plot the results per encoding
    for encoding in ['tabular', 'sequence', 'graph_embedding', 'graph']:
        for model in ['linear_regression', 'random_forest', 'regression_tree', 'mlp']:
            df = results_df[(results_df['encoding'] == encoding) & (results_df['model'] == model)]
            # Skip empty results
            if df.empty:
                continue

            # Plot the results and save them
            g = sns.catplot(df, x='features', y='mean_absolute_error', hue='flattening', kind='bar')
            g.set_ylabels('Mean absolute error')
            plt.savefig(os.path.join(ROOT_DIR, 'reports', dataset['filename'], f'{encoding}_{model}.svg'))
            plt.savefig(os.path.join(ROOT_DIR, 'reports', dataset['filename'], f'{encoding}_{model}.pdf'))
            plt.close()

    df = results_df[(results_df['model'] == 'random_forest')]

    # Add a column with the encoding and model
    encoding_mapping = {
        'tabular': 'Tabular',
        'sequence': 'LSTM',
        'graph_embedding': 'Graph embedding',
        'graph': 'GCN',
    }
    df['encoding'] = df['encoding'].map(encoding_mapping)

    # Plot the results and save them
    g = sns.relplot(df, x='features', y='mean_absolute_error', hue='encoding', col='flattening', kind='scatter')
    g.set_ylabels('Mean absolute error')
    plt.savefig(os.path.join(ROOT_DIR, 'reports', dataset['filename'], f'all.svg'))
    plt.savefig(os.path.join(ROOT_DIR, 'reports', dataset['filename'], f'all.pdf'))
    plt.close()

    print('Visualizing results')


def calculate_improvements():
    dataset = get_selected_dataset()

    results_df = pd.read_csv(os.path.join(ROOT_DIR, 'models', dataset['filename'], 'results_combined.csv'))
    mean_df = pd.read_csv(os.path.join(ROOT_DIR, 'models', dataset['filename'], 'results_mean.csv'))

    # Add the feature names to the results
    feature_mapping = {
        2: 'Baseline',
        4: 'Basic',
        10: 'Basic + Enhanced',
        17: 'All',
    }

    # Clean up the flattening column
    results_df['flattening'] = results_df['flattening'].map(lambda x: 'Compound object type' if x == 'ALL' else 'None')

    # Calculate the improvements with increasing features with respect to the baseline for each encoding and flattening
    for encoding in ['tabular', 'sequence', 'graph_embedding', 'graph']:
        for model in ['linear_regression', 'random_forest', 'regression_tree', 'mlp']:
            df = results_df[(results_df['encoding'] == encoding) & (results_df['model'] == model)].copy()
            # Skip empty results
            if df.empty:
                continue

            baseline = mean_df[(mean_df['encoding'] == encoding)]['mean_absolute_error'].values[0]
            # Calculate the improvement with respect to the baseline in percent
            df['improvement'] = (baseline - df['mean_absolute_error']) / baseline * 100
            # Calculate the difference between the two flattening methods
            df = df.sort_values(by=['features', 'flattening'])
            df['x_absolute_difference'] = (df['improvement'] - df['improvement'].shift(1))
            df['mae_difference'] = (df['mean_absolute_error'] - df['mean_absolute_error'].shift(1)) * -1
            df['x_relative_difference'] = (df['mae_difference'] / df['mean_absolute_error'].shift(1)) * 100
            # Only keep the difference for no flattening
            df['x_absolute_difference'] = df['x_absolute_difference'].where(df['flattening'] == 'None')
            df['x_relative_difference'] = df['x_relative_difference'].where(df['flattening'] == 'None')

            # Create latex table
            df = df[['flattening', 'features', 'improvement', 'x_absolute_difference', 'x_relative_difference']]

            # Stack the difference column
            df = df.set_index(['flattening', 'features'])
            df = df.stack().reset_index()

            df = df.rename(columns={
                'flattening': 'Flattening',
                'features': 'Features',
                'improvement': 'Improvement',
            })
            df[0] = df[0].map(lambda x: str(f'{x:.2f}') + '\%')
            print(df)
            # Map the feature numbers to the feature names
            df = df.pivot(index=['Flattening', 'level_2'], columns='Features')
            to_latex_table(df, filename=f'{encoding}_{model}_improvements.tex',
                           caption=f'Improvements for {encoding}_{model} encoding',
                           label=f'tab:improvements_{encoding}_{model}')


if __name__ == '__main__':
    visualize_results()
    calculate_improvements()
