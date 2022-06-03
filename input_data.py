import pickle as pkl

def load_data(dataset):
    file_name = f"data/{dataset}.pkl"

    with open(file_name, "rb") as f:
        graphs = pkl.load(f)
    return graphs
