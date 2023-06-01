import logging
import os

import pandas as pd
import tensorflow as tf
import networkx as nx
import numpy as np
from karateclub import GL2Vec
from tqdm import tqdm

from ocpa.algo.predictive_monitoring.obj import Feature_Storage

os.environ['DGLBACKEND'] = 'tensorflow'
import dgl


def reset_node_indices(g: nx.DiGraph) -> nx.DiGraph:
    mapping = {n: i for i, n in enumerate(sorted(g.nodes()))}
    return nx.relabel_nodes(g, mapping)


def _convert_feature_graph_to_networkx_graph(feature_graph: Feature_Storage.Feature_Graph,
                                             target: tuple = None) -> nx.DiGraph:
    """
    Convert a feature graph to a networkx graph

    :param feature_graph: The feature graph to convert
    :type feature_graph: Feature_Storage.Feature_Graph
    :param target: The target feature
    :type target: tuple

    :return: The converted networkx graph
    :rtype: nx.DiGraph
    """
    graph = nx.DiGraph(incoming_graph_data=None, **feature_graph.attributes)
    graph.add_nodes_from((node.event_id, {k: v for k, v in node.attributes.items() if k != target})
                         for node in feature_graph.nodes)
    graph.add_edges_from((edge.source, edge.target, edge.attributes) for edge in feature_graph.edges)
    return graph


def convert_networkx_graph_to_dgl_graph(graph: nx.DiGraph) -> dgl.DGLGraph:
    """
    Convert a networkx graph to a dgl graph

    :param graph: The networkx graph to convert
    :type graph: nx.DiGraph

    :return: The converted dgl graph
    :rtype: dgl.DGLGraph
    """
    # reset node ids
    graph = reset_node_indices(graph)
    dgl_graph = dgl.graph(data=([e[0] for e in graph.edges], [e[1] for e in graph.edges]),
                          num_nodes=len(graph.nodes))

    features = []
    for node in graph.nodes(data=True):
        # assert i == len(features), "Node ids are not consecutive."
        features.append([v for k, v in node[1].items()])
    dgl_graph.ndata['features'] = tf.constant(features, dtype=tf.float32)

    return dgl_graph


def convert_to_nx_graphs(g: Feature_Storage.Feature_Graph, k: int, target: tuple,
                         from_start=False, include_last=True):
    return_graphs = []
    target_values = []

    # sort nodes on event time (through the event id)
    sorted_idxs = [n.event_id for n in g.nodes]
    sorted_idxs.sort()
    end_index = 0 if from_start else len(sorted_idxs) - k
    if not include_last:
        end_index -= 1

    # to networkx graph
    nx_graph = nx.Graph()
    for edge in g.edges:
        nx_graph.add_edge(edge.source, edge.target)
    nx.set_node_attributes(nx_graph, {n.event_id: n.attributes for n in g.nodes})

    # extract subgraphs
    for start in range(0, end_index + 1):
        subgraph = nx.subgraph(nx_graph, sorted_idxs[start:start + k]).copy()
        for node in subgraph.nodes():
            val = subgraph.nodes()[node][target]
            del subgraph.nodes()[node][target]

            if node == sorted_idxs[start + k - 1]:
                target_values.append(val)

        indexed_subgraph = nx.convert_node_labels_to_integers(subgraph)

        return_graphs.append(indexed_subgraph)

    return return_graphs, target_values


def generate_graph_prefixes(feature_storage: Feature_Storage, target: tuple, trace_length: int, index_list='all',
                            undirected: bool = False) -> tuple[list[nx.DiGraph], list[float]]:
    """
    Generate graph prefixes from the feature storage

    :param feature_storage: Feature storage to construct a graph encoding from.
    :type feature_storage: :class:`Feature Storage <ocpa.algo.predictive_monitoring.obj.Feature_Storage>`
    :param target: The target feature
    :type target: tuple
    :param trace_length: The length of the trace
    :type trace_length: int
    :param index_list: list of indices to be encoded as sequences. Default is "all"
    :type index_list: "all" or list(int)
    :param undirected: Whether to convert the graph to an undirected graph
    :type undirected: bool

    :return: Tuple of all graph prefixes and their corresponding target values
    :rtype: tuple[list[nx.DiGraph], list[float]]
    """
    if index_list == "all":
        fgs = feature_storage.feature_graphs
    else:
        fgs = list(filter(lambda x: x.pexec_id in index_list, feature_storage.feature_graphs))

    fgs.sort(key=lambda x: x.pexec_id)

    sub_graphs = []
    target_values = []
    for g in tqdm(fgs):
        result = _generate_graph_prefix((g, target, trace_length))
        node_data, edge_data, target_value = result
        for i in range(len(node_data)):
            if undirected:
                graph = nx.Graph(incoming_graph_data=None)
            else:
                graph = nx.DiGraph(incoming_graph_data=None)
            graph.add_nodes_from(node_data[i])
            graph.add_edges_from(edge_data[i])
            sub_graphs.append(graph)
            target_values.append(target_value[i])

    return sub_graphs, target_values


def _generate_graph_prefix(params: tuple[Feature_Storage.Feature_Graph, tuple, int]):
    feature_graph, target, trace_length = params

    node_data = []
    edge_data = []
    target_values = []

    # sort nodes on event time (through the event id)
    event_ids = [n.event_id for n in feature_graph.nodes]
    event_ids.sort()
    # Generate all subgraphs
    for i in range(trace_length - 1, len(event_ids)):
        subgraph_event_ids = event_ids[max(0, i - trace_length + 1): i + 1]

        node_data.append(list((node.event_id, {k: v for k, v in node.attributes.items() if k != target})
                              for node in feature_graph.nodes if node.event_id in subgraph_event_ids))
        edge_data.append(list((edge.source, edge.target) for edge in feature_graph.edges
                              if edge.source in subgraph_event_ids and edge.target in subgraph_event_ids))
        target_values.append(feature_graph.get_node_from_event_id(event_ids[i]).attributes[target])
    return node_data, edge_data, target_values


def graph_embedding_with_features(feature_storage: Feature_Storage, target: tuple, model_type: str = 'FGSD',
                                  trace_length: int = 4) -> tuple[
    np.ndarray, list[float], np.ndarray, list[float], any]:
    """
    Generate graph embeddings from the feature storage

    :param feature_storage: Feature storage to construct a graph encoding from.
    :type feature_storage: :class:`Feature Storage <ocpa.algo.predictive_monitoring.obj.Feature_Storage>`
    :param target: The target feature
    :type target: tuple
    :param model_type: The graph embedding model_type to use
    :type model_type: str
    :param trace_length: The length of the trace
    :type trace_length: int

    :return: Tuple of the graph embeddings, the target values and the model
    :rtype: tuple[np.ndarray, list[float], np.ndarray, list[float], GL2Vec]
    """
    # Generate graph prefixes
    logging.info('Generating graph prefixes')

    undirected = False
    if model_type in ['FGSD', 'WaveletCharacteristic']:
        undirected = True

    graphs_train, y_train = generate_graph_prefixes(feature_storage, target, trace_length, undirected=undirected,
                                                    index_list=feature_storage.training_indices)
    graphs_test, y_test = generate_graph_prefixes(feature_storage, target, trace_length, undirected=undirected,
                                                  index_list=feature_storage.test_indices)
    logging.info(f'Generated {len(graphs_train)} training graph prefixes and {len(graphs_test)} test graph prefixes')

    # reset node indices of the graph prefixes
    # this is necessary because karateclub expects the node indices to be consecutive and starting at 0
    logging.info('Resetting node indices')
    graphs_train = [reset_node_indices(graph) for graph in graphs_train]
    graphs_test = [reset_node_indices(graph) for graph in graphs_test]

    # Select the model
    import multiprocessing
    cpu_count = round(multiprocessing.cpu_count() * 0.8)
    if model_type == 'GL2Vec':
        from karateclub import GL2Vec
        model = GL2Vec(workers=cpu_count)
    elif model_type == 'FGSD':
        from karateclub import FGSD
        model = FGSD()
    elif model_type == 'WaveletCharacteristic':
        from karateclub import WaveletCharacteristic
        model = WaveletCharacteristic()
    else:
        raise ValueError(f"Model {model_type} not supported. Supported models are: GL2Vec, FGSD, WaveletCharacteristic")

    # Fit the model
    logging.info('Fitting the graph embedding model')
    model.fit(graphs_train)
    X_train_graphs = model.get_embedding()
    X_test_graphs = model.infer(graphs_test)

    # Extract the features from the graphs
    logging.info('Extracting features from the graphs')

    def _extract_features(g: nx.DiGraph) -> np.ndarray:
        return pd.DataFrame(g.nodes(data=True)._nodes, columns=list(range(trace_length))).sort_index().values.T.ravel()

    X_train = np.concatenate([X_train_graphs, np.array([_extract_features(g) for g in graphs_train])], axis=1)
    X_test = np.concatenate([X_test_graphs, np.array([_extract_features(g) for g in graphs_test])], axis=1)

    logging.info(f'Extracted {X_train.shape[1]} features from the graphs')
    # Return the embedding, the target values and the model
    return X_train, y_train, X_test, y_test, model
