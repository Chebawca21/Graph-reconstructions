from __future__ import division
from __future__ import print_function

import time

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp

from graphite_optimizer import OptimizerVAE
from input_data import *
from graphite_model import *
from graphite_preprocessing import *

def scale_graph(graph):
    for i in range(20):
        if not graph.has_node(i):
            graph.add_node(i)
    return graph

MAXNODES = 20

def graphite_train(learning_rate=0.01, epochs=3, hidden1=32, hidden2=16, hidden3=32, dropout=0., auto_scalar=0., dataset="erdos_renyi", features=0, seed=21, vae=1):

    # Settings
    tf.disable_v2_behavior()

    np.random.seed(seed=seed)
    tf.compat.v1.set_random_seed(seed)

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

    model = GCNModelFeedback(placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2, hidden3, vae, auto_scalar)

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

    with tf.name_scope('optimizer'):
        opt = OptimizerVAE(preds=model.reconstructions,
                        labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                        model=model, num_nodes=num_nodes,
                        pos_weight=pos_weight,
                        norm=norm, lr=learning_rate, vae=vae)

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