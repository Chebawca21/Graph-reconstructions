import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


# def parse_index_file(filename):
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index


# def load_data(dataset):
#     # load the data: x, tx, allx, graph
#     names = ['x', 'tx', 'allx', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#     x, tx, allx, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
#     test_idx_range = np.sort(test_idx_reorder)

#     if dataset == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended

#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#     return adj, features

def load_data(dataset):
    file_name = f"data/{dataset}.pkl"

    with open(file_name, "rb") as f:
        graphs = pkl.load(f)
    return graphs

# graphs = load_data("erdos_renyi")
# np.random.shuffle(graphs)
# graph = nx.from_dict_of_dicts(graphs[0])
# print(graph)

# num_nodes = graph.number_of_nodes()
# for i in range(20 - num_nodes):
#     graph.add_node(i+num_nodes)

# print(graph)
# adj = nx.adjacency_matrix(graph)


