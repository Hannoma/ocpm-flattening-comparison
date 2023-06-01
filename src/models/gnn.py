import logging
import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from ocpa.algo.predictive_monitoring.obj import Feature_Storage
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from definitions import ROOT_DIR
from src.encoding.graphs import generate_graph_prefixes, convert_networkx_graph_to_dgl_graph
from src.models.lstm import training_loop
from src.models.train_model import calculate_scores

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['DGLBACKEND'] = 'tensorflow'

from dgl.nn import GraphConv
import dgl


# custom data loader for yielding batches of graphs
# from https://github.com/niklasadams/OCELFeatureExtractionExperiments/blob/main/gnn_utils.py
class GraphDataLoader(tf.keras.utils.Sequence):
    def __init__(self, graph_list, graph_labels, batch_size, shuffle=True, add_self_loop=False, make_bidirected=False,
                 on_gpu=False):
        self.graph_list = graph_list
        self.graph_labels = graph_labels
        self.batch_size = batch_size
        self.add_self_loop = add_self_loop
        self.make_bidirected = make_bidirected
        self.indices = np.arange(0, len(graph_list))
        self.on_gpu = on_gpu
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.graph_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        graph_batch = [self.graph_list[i] for i in batch_indices]
        if self.add_self_loop:
            graph_batch = [dgl.add_self_loop(g) for g in graph_batch]
        if self.make_bidirected:
            with tf.device('CPU:0'):
                graph_batch = [dgl.to_bidirected(g, copy_ndata=True) for g in graph_batch]
        if self.on_gpu:
            graph_batch = [g.to('GPU:0') for g in graph_batch]
        dgl_batch = dgl.batch(graph_batch)

        if not np.all(dgl_batch.in_degrees() > 0):
            print('WARNING: 0-in-degree nodes found!')

        labels_batch = [self.graph_labels[i] for i in batch_indices]
        labels_batch = tf.stack(labels_batch, axis=0)
        if self.on_gpu:
            labels_batch = labels_batch.to('GPU:0')

        return dgl_batch, labels_batch


class GCN_Model(tf.keras.Model):
    def __init__(self, in_feats, h_feats):
        super().__init__()
        self.n_layers = 2
        self.n_neurons = h_feats
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, g, training=None, mask=None):
        h = self.conv1(g, g.ndata['features'])
        h = tf.keras.activations.relu(h)
        h = self.conv2(g, h)
        h = tf.keras.activations.relu(h)
        x = tf.reshape(h, (int(h.shape[0] / 4), (4 * self.n_neurons)))
        out = self.dense(x)
        return out

    def my_name(self):
        return f'GCN_{self.n_layers}_{self.n_neurons}'


def training_loop(model: GCN_Model, train_dataloader: GraphDataLoader, val_dataloader: GraphDataLoader,
                  num_epochs: int, dataset_name: str):
    # print(model.summary())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    early_stopping = EarlyStopping(patience=42)
    model_checkpoint = ModelCheckpoint(
        os.path.join(ROOT_DIR, 'models', dataset_name, model.my_name() + '_weights_best.h5'),
        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    callbacks = [early_stopping, model_checkpoint, lr_reducer]

    # train the model (maxlen)
    model.fit(train_dataloader, validation_data=val_dataloader, callbacks=callbacks, epochs=num_epochs, batch_size=32,
              use_multiprocessing=True, workers=12)

    # saving model to file
    model_json = model.to_json()
    with open(os.path.join(ROOT_DIR, 'models', dataset_name, model.name() + '.json'), "w") as json_file:
        json_file.write(model_json)
    return model


def custom_training_loop(model: GCN_Model, train_dataloader: GraphDataLoader, val_dataloader: GraphDataLoader,
                         num_epochs: int, dataset_name: str):
    # print(model.summary())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    iter_idx = np.arange(0, train_dataloader.__len__())
    loss_history = []
    val_loss_history = []
    step_losses = []
    for e in range(num_epochs):
        print('Running epoch:', e)
        np.random.shuffle(iter_idx)
        current_loss = step = 0
        for batch_id in tqdm(iter_idx):
            step += 1
            dgl_batch, label_batch = train_dataloader.__getitem__(batch_id)
            with tf.GradientTape() as tape:
                pred = model(dgl_batch, dgl_batch.ndata['features'])
                loss = loss_function(label_batch, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            step_losses.append(loss.numpy())
            current_loss += loss.numpy()
            # if (step % 100 == 0): print('Loss: %s'%((current_loss / step)))
            loss_history.append(current_loss / step)
        val_predictions, val_labels = predict_gnn(model, val_dataloader)
        val_loss = tf.keras.metrics.mean_squared_error(np.squeeze(val_labels), np.squeeze(val_predictions)).numpy()
        print('    Validation MSE GNN:', val_loss)
        if len(val_loss_history) < 1 or val_loss < np.min(val_loss_history):
            model.save_weights(os.path.join(ROOT_DIR, 'models', dataset_name, model.my_name() + '_weights_best.tf'))
            print('    GNN checkpoint saved.')
        val_loss_history.append(val_loss)

    # restore weights from best epoch
    cp_status = model.load_weights(os.path.join(ROOT_DIR, 'models', dataset_name, model.my_name() + '_weights_best.tf'))
    cp_status.assert_consumed()

    return model


def train_model_with_gnn(feature_storage: Feature_Storage, target: tuple, dataset_name: str, n_layers: int = 2,
                         n_neurons: int = 32, num_epochs: int = 50) -> (GCN_Model, list):
    # Generate graph prefixes
    logging.info('Generating graph prefixes')
    graphs_train, y_train = generate_graph_prefixes(feature_storage, target, trace_length=4,
                                                    index_list=feature_storage.training_indices)
    graphs_test, y_test = generate_graph_prefixes(feature_storage, target, trace_length=4,
                                                  index_list=feature_storage.test_indices)
    logging.info(f'Generated {len(graphs_train)} training graph prefixes and {len(graphs_test)} test graph prefixes')

    # convert to dgl format
    logging.info(f'Converting graph prefixes to DGL format')
    X_training = list(map(convert_networkx_graph_to_dgl_graph, graphs_train))
    X_test = list(map(convert_networkx_graph_to_dgl_graph, graphs_test))
    # convert to numpy array
    y_training = np.asarray(y_train, dtype='float32')
    y_test = np.asarray(y_test, dtype='float32')

    # split training data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, random_state=42)
    train_dl = GraphDataLoader(graph_list=X_train, graph_labels=y_train, batch_size=32, on_gpu=True, add_self_loop=True)
    val_dl = GraphDataLoader(graph_list=X_val, graph_labels=y_val, batch_size=32, on_gpu=True, add_self_loop=True)
    test_dl = GraphDataLoader(graph_list=X_test, graph_labels=y_test, batch_size=32, on_gpu=True, add_self_loop=True)

    # build model
    logging.info(f'Building model')
    model = GCN_Model(in_feats=len(feature_storage.event_features) - 1, h_feats=n_neurons)
    model.to('GPU:0')

    # compile model
    model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])

    # train model
    logging.info(f'Training model')
    model = custom_training_loop(model, train_dl, val_dl, num_epochs=num_epochs, dataset_name=dataset_name)

    # evaluate model
    logging.info(f'Evaluating model')
    y_pred, y_true = predict_gnn(model, test_dl)
    scores = calculate_scores(y_pred, y_true, verbose=True)

    return (), scores


def predict_gnn(model: GCN_Model, data_loader: GraphDataLoader):
    predictions = []
    labels = []
    for batch_id in tqdm(range(data_loader.__len__())):
        dgl_batch, label_batch = data_loader.__getitem__(batch_id)
        pred = model(dgl_batch, dgl_batch.ndata['features']).numpy()
        predictions.append(pred)
        labels.append(label_batch.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return predictions, labels
