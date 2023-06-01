import logging

import numpy as np

from definitions import WORKER_COUNT
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error

from src.encoding.graphs import graph_embedding_with_features, generate_graph_prefixes
from src.encoding.sequential import generate_trace_prefixes
from src.encoding.tabular import tabular_encoding


def train_model_with_feature_storage(feature_storage: Feature_Storage, target: tuple, model_type: str,
                                     model_params: dict = None) -> tuple:
    """Train a model on the given feature storage.

    :param feature_storage: Feature storage to train on
    :type feature_storage: :class:`Feature Storage <ocpa.algo.predictive_monitoring.obj.Feature_Storage>`
    :param target: Name of the target attribute
    :type target: str
    :param model_type: Type of model to train
    :type model_type: str
    :param model_params: Parameters for the model
    :type model_params: dict

    :return: Trained model
    """
    if model_params is None:
        model_params = {}

    # Get the training and test data
    X_train, y_train, X_test, y_test = tabular_encoding(feature_storage, target)

    return _train_model(X_train, y_train, X_test, y_test, model_type, model_params)


def train_model_with_graph_embedding(feature_storage: Feature_Storage, target: tuple, model_type: str,
                                     model_params: dict = None) -> tuple:
    """Train a model with the given graph embedding.

    :param feature_storage: Feature storage to train on
    :type feature_storage: :class:`Feature Storage <ocpa.algo.predictive_monitoring.obj.Feature_Storage>`
    :param target: Name of the target attribute
    :type target: str
    :param model_type: Type of model to train
    :type model_type: str
    :param model_params: Parameters for the model
    :type model_params: dict
    """

    X_train, y_train, X_test, y_test, _ = graph_embedding_with_features(feature_storage, target)

    return _train_model(X_train, y_train, X_test, y_test, model_type, model_params)


def _train_model(X_train, y_train, X_test, y_test, model_types, model_params) -> tuple:
    if type(model_types) == str:
        model_types = [model_types]

    results = {}
    for model_type in model_types:
        logging.info(f'Training {model_type} model')
        # Get the specified model
        if model_type == 'regression_tree':
            from sklearn.tree import DecisionTreeRegressor
            regressor = DecisionTreeRegressor()
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            regressor = RandomForestRegressor(n_jobs=WORKER_COUNT, n_estimators=250)
        elif model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression(n_jobs=WORKER_COUNT)
        elif model_type == 'mlp':
            from sklearn.neural_network import MLPRegressor
            regressor = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=200, verbose=True)
        else:
            raise ValueError('Please specify a valid model type. Currently supported: '
                             'regression_tree, random_forest, linear_regression, mlp')

        # fit the model
        regressor.fit(X_train, y_train)

        # score the model
        # Predict the target variable
        y_pred = regressor.predict(X_test)
        scores = calculate_scores(y_pred, y_test, verbose=True)
        results[model_type] = scores

    return (), results


def calculate_scores(y_pred, y_true, verbose=False) -> dict[str, float]:
    """Calculate the scores for the given predictions.

    :param y_pred: Predictions
    :type y_pred: np.ndarray
    :param y_true: True values
    :type y_true: np.ndarray
    :param verbose: Print the scores
    :type verbose: bool

    :return: Scores
    :rtype: dict[str, float]
    """

    score = r2_score(y_true=y_true, y_pred=y_pred)
    error = max_error(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)

    if verbose:
        print(f'R2-score: {score}')
        print(f'Max error: {error}')
        print(f'Mean absolute error: {mae}')
        print(f'Mean squared error: {mse}')
        print(f'Root mean squared error: {np.sqrt(mse)}')

    return {
        'r2_score': score,
        'max_error': error,
        'mean_absolute_error': mae,
        'mean_squared_error': mse,
    }


def train_baseline_model(feature_storage: Feature_Storage, target, encoding):
    if encoding == 'tabular':
        X_train, y_train, X_test, y_test = tabular_encoding(feature_storage, target)
    elif encoding == 'graph_embedding':
        graphs_train, y_train = generate_graph_prefixes(feature_storage, target, trace_length=4,
                                                        index_list=feature_storage.training_indices)
        graphs_test, y_test = generate_graph_prefixes(feature_storage, target, trace_length=4,
                                                      index_list=feature_storage.test_indices)
        logging.info(f'Generated {len(graphs_train)} training graph prefixes and {len(graphs_test)} test graph prefixes')
    elif encoding == 'sequence':
        # generate trace prefixes
        X_train, y_train = generate_trace_prefixes(feature_storage=feature_storage,
                                                   target=target,
                                                   min_trace_length=4,
                                                   max_trace_length=4,
                                                   index_list=feature_storage.training_indices)
        X_test, y_test = generate_trace_prefixes(feature_storage=feature_storage,
                                                 target=target,
                                                 min_trace_length=4,
                                                 max_trace_length=4,
                                                 index_list=feature_storage.test_indices)
        logging.info(f'Generated {len(X_train)} training trace prefixes and {len(X_test)} test trace prefixes')
    elif encoding == 'graph':
        # Generate graph prefixes
        graphs_train, y_train = generate_graph_prefixes(feature_storage, target, trace_length=4,
                                                        index_list=feature_storage.training_indices)
        graphs_test, y_test = generate_graph_prefixes(feature_storage, target, trace_length=4,
                                                      index_list=feature_storage.test_indices)
        logging.info(f'Generated {len(graphs_train)} training graph prefixes and {len(graphs_test)} test graph prefixes')
    else:
        raise ValueError(f'Unknown encoding type: {encoding}')

    mean = np.mean(y_train)
    y_pred = np.full(len(y_test), mean)
    scores = calculate_scores(y_pred, y_test, verbose=True)
    return None, scores
