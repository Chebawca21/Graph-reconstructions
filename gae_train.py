import time
import os

from more_itertools import sample

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx

from gae_optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from gae_model import GCNModelAE, GCNModelVAE
from gae_preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple

def scale_graph(graph):
    for i in range(20):
        if not graph.has_node(i):
            graph.add_node(i)
    return graph

MAXNODES = 20

def gae_train(learning_rate=0.01, epochs=5, hidden1=32, hidden2=16, dropout=0., model="gcn_ae", dataset="erdos_renyi", features=0, seed=21):

    # Settings
    tf.disable_v2_behavior()

    np.random.seed(seed=seed)
    tf.compat.v1.set_random_seed(seed)

    model_str = model
    dataset_str = dataset

    # Load data
    graphs = load_data(dataset_str)

    # Split dataset to train, val and test
    np.random.shuffle(graphs)
    dataset_size = int(len(graphs) / 3)
    train_dataset = graphs[:dataset_size]
    val_dataset = graphs[dataset_size:dataset_size*2]
    test_dataset = graphs[dataset_size*2:]

    if features == 0:
        features = sp.identity(MAXNODES)  # featureless

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = MAXNODES

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]


    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero, hidden1, hidden2)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2)

    # Count average values for pos_weight and norm
    pos_weight = 0.0
    norm = 0.0
    for graph in graphs:
        adj = nx.from_dict_of_dicts(graph)
        adj = nx.adjacency_matrix(adj)

        pos_weight += float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm += adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    pos_weight = pos_weight / len(graphs)
    norm = norm / len(graphs)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            pos_weight=pos_weight,
                            norm=norm, lr=learning_rate)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm, lr=learning_rate)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_loss(dataset):
        cost = 0.0
        for graph in dataset:
            # Transforming to adjacency matrix
            g = nx.from_dict_of_dicts(graph)
            g = scale_graph(g)
            adj = nx.adjacency_matrix(g)

            # Some preprocessing
            adj_norm = preprocess_graph(adj)
            adj_label = adj + sp.eye(adj.shape[0])
            adj_label = sparse_to_tuple(adj_label)

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: 0})

            if model == "gcn_vae":
                cost += sess.run(opt.log_lik, feed_dict=feed_dict)
            else:
                cost += sess.run(opt.cost, feed_dict=feed_dict)
        
        return cost

    # Train model
    for epoch in range(epochs):

        cost = 0.0
        accuracy = 0.0
        t = time.time()
        for graph in train_dataset:
            # Transforming to adjacency matrix
            g = nx.from_dict_of_dicts(graph)
            g = scale_graph(g)
            adj = nx.adjacency_matrix(g)

            # Some preprocessing
            adj_norm = preprocess_graph(adj)

            adj_label = adj + sp.eye(adj.shape[0])
            adj_label = sparse_to_tuple(adj_label)

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            cost += outs[1]
            accuracy += outs[2]

        val_cost = get_loss(val_dataset)

        avg_accuracy = accuracy / len(train_dataset) 
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cost),
            "train_acc=", "{:.5f}".format(avg_accuracy), "val_loss=", "{:.5f}".format(val_cost),
            "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    test_cost = get_loss(test_dataset)
    print('Test loss: ' + str(test_cost))

    return test_cost