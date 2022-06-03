import networkx as nx
import numpy.random as rnd
import pickle as pkl

NGRAPHS = 300
MINVERTICIES = 10
MAXVERTICIES = 20
SEED = 21

rnd.seed(seed=SEED)

def generate_erdos_renyi():
    p = 0.5
    graphs = []
    for _ in range(NGRAPHS):
        n = rnd.randint(MINVERTICIES, MAXVERTICIES)
        g = nx.erdos_renyi_graph(n, p, seed=SEED)
        graphs.append(nx.to_dict_of_dicts(g))
    
    with open('data/erdos_renyi.pkl', 'wb') as f:
        pkl.dump(graphs, f)

def generate_ego_net():
    gradius = 0.6
    node = 0
    eradius = 10
    graphs = []
    for _ in range(NGRAPHS):
        n = rnd.randint(MINVERTICIES, MAXVERTICIES)
        g = nx.random_geometric_graph(n, gradius, seed=SEED)
        g = nx.ego_graph(g, node, radius=eradius)
        graphs.append(nx.to_dict_of_dicts(g))
    
    with open('data/ego_net.pkl', 'wb') as f:
        pkl.dump(graphs, f)

def generate_random_regular():
    graphs = []
    for _ in range(NGRAPHS):
        n = rnd.randint(MINVERTICIES, MAXVERTICIES)
        d = rnd.randint(3, n-1)
        if n * d % 2 == 1:
            if d < n-1:
                d = d + 1
            else:
                d = d - 1
        g = nx.random_regular_graph(d, n, seed=SEED)
        graphs.append(nx.to_dict_of_dicts(g))
    
    with open('data/random_regular.pkl', 'wb') as f:
        pkl.dump(graphs, f)

def generate_random_geometric():
    radius = 0.5
    graphs = []
    for _ in range(NGRAPHS):
        n = rnd.randint(MINVERTICIES, MAXVERTICIES)
        g = nx.random_geometric_graph(n, radius, seed=SEED)
        graphs.append(nx.to_dict_of_dicts(g))
    
    with open('data/random_geometric.pkl', 'wb') as f:
        pkl.dump(graphs, f)

def generate_random_powerlaw_tree():
    graphs = []
    for _ in range(NGRAPHS):
        n = rnd.randint(MINVERTICIES, MAXVERTICIES)
        g = nx.random_powerlaw_tree(n, tries=1000, seed=SEED)
        graphs.append(nx.to_dict_of_dicts(g))
    
    with open('data/random_powerlaw_tree.pkl', 'wb') as f:
        pkl.dump(graphs, f)

def generate_barabasi_albert():
    m = 3
    graphs = []
    for _ in range(NGRAPHS):
        n = rnd.randint(MINVERTICIES, MAXVERTICIES)
        g = nx.barabasi_albert_graph(n, m, seed=SEED)
        graphs.append(nx.to_dict_of_dicts(g))
    
    with open('data/barabasi_albert.pkl', 'wb') as f:
        pkl.dump(graphs, f)

generate_erdos_renyi()
generate_ego_net()
generate_random_regular()
generate_random_geometric()
generate_random_powerlaw_tree()
generate_barabasi_albert()