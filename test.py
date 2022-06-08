from gae_train import gae_train
from graphite_train import graphite_train

seeds = [21]
epochs = 100

gae_er, gae_en, gae_rr, gae_rg, gae_pt, gae_ba = [], [], [], [], [], []
vgae_er, vgae_en, vgae_rr, vgae_rg, vgae_pt, vgae_ba = [], [], [], [], [], []
graphite_ae_er, graphite_ae_en, graphite_ae_rr, graphite_ae_rg, graphite_ae_pt, graphite_ae_ba = [], [], [], [], [], []
graphite_vae_er, graphite_vae_en, graphite_vae_rr, graphite_vae_rg, graphite_vae_pt, graphite_vae_ba = [], [], [], [], [], []

for seed in seeds:
    gae_er.append(gae_train(epochs=epochs, model="gcn_ae", dataset="erdos_renyi", seed=seed))
    gae_en.append(gae_train(epochs=epochs, model="gcn_ae", dataset="ego_net", seed=seed))
    gae_rr.append(gae_train(epochs=epochs, model="gcn_ae", dataset="random_regular", seed=seed))
    gae_rg.append(gae_train(epochs=epochs, model="gcn_ae", dataset="random_geometric", seed=seed))
    gae_pt.append(gae_train(epochs=epochs, model="gcn_ae", dataset="random_powerlaw_tree", seed=seed))
    gae_ba.append(gae_train(epochs=epochs, model="gcn_ae", dataset="barabasi_albert", seed=seed))

    vgae_er.append(gae_train(epochs=epochs, model="gcn_vae", dataset="erdos_renyi", seed=seed))
    vgae_en.append(gae_train(epochs=epochs, model="gcn_vae", dataset="ego_net", seed=seed))
    vgae_rr.append(gae_train(epochs=epochs, model="gcn_vae", dataset="random_regular", seed=seed))
    vgae_rg.append(gae_train(epochs=epochs, model="gcn_vae", dataset="random_geometric", seed=seed))
    vgae_pt.append(gae_train(epochs=epochs, model="gcn_vae", dataset="random_powerlaw_tree", seed=seed))
    vgae_ba.append(gae_train(epochs=epochs, model="gcn_vae", dataset="barabasi_albert", seed=seed))

    graphite_ae_er.append(graphite_train(epochs=epochs, vae=0, dataset="erdos_renyi", seed=seed))
    graphite_ae_en.append(graphite_train(epochs=epochs, vae=0, dataset="ego_net", seed=seed))
    graphite_ae_rr.append(graphite_train(epochs=epochs, vae=0, dataset="random_regular", seed=seed))
    graphite_ae_rg.append(graphite_train(epochs=epochs, vae=0, dataset="random_geometric", seed=seed))
    graphite_ae_pt.append(graphite_train(epochs=epochs, vae=0, dataset="random_powerlaw_tree", seed=seed))
    graphite_ae_ba.append(graphite_train(epochs=epochs, vae=0, dataset="barabasi_albert", seed=seed))

    graphite_vae_er.append(graphite_train(epochs=epochs, vae=1, dataset="erdos_renyi", seed=seed))
    graphite_vae_en.append(graphite_train(epochs=epochs, vae=1, dataset="ego_net", seed=seed))
    graphite_vae_rr.append(graphite_train(epochs=epochs, vae=1, dataset="random_regular", seed=seed))
    graphite_vae_rg.append(graphite_train(epochs=epochs, vae=1, dataset="random_geometric", seed=seed))
    graphite_vae_pt.append(graphite_train(epochs=epochs, vae=1, dataset="random_powerlaw_tree", seed=seed))
    graphite_vae_ba.append(graphite_train(epochs=epochs, vae=1, dataset="barabasi_albert", seed=seed))

print("\n\n")
print("======================================")

for i in range(len(seeds)):
    print("Erdos-Renyi")
    print("GAE: ", gae_er[i])
    print("Graphite-AE: ", graphite_ae_er[i])
    print("VGAE: ", vgae_er[i])
    print("Graphite-VAE: ", graphite_vae_er[i])
    print("\n")

    print("Ego")
    print("GAE: ", gae_en[i])
    print("Graphite-AE: ", graphite_ae_en[i])
    print("VGAE: ", vgae_en[i])
    print("Graphite-VAE: ", graphite_vae_en[i])
    print("\n")

    print("Regular")
    print("GAE: ", gae_rr[i])
    print("Graphite-AE: ", graphite_ae_rr[i])
    print("VGAE: ", vgae_rr[i])
    print("Graphite-VAE: ", graphite_vae_rr[i])
    print("\n")

    print("Geometric")
    print("GAE: ", gae_rg[i])
    print("Graphite-AE: ", graphite_ae_rg[i])
    print("VGAE: ", vgae_rg[i])
    print("Graphite-VAE: ", graphite_vae_rg[i])
    print("\n")

    print("Power Law")
    print("GAE: ", gae_pt[i])
    print("Graphite-AE: ", graphite_ae_pt[i])
    print("VGAE: ", vgae_pt[i])
    print("Graphite-VAE: ", graphite_vae_pt[i])
    print("\n")

    print("Barabasi-Albert")
    print("GAE: ", gae_ba[i])
    print("Graphite-AE: ", graphite_ae_ba[i])
    print("VGAE: ", vgae_ba[i])
    print("Graphite-VAE: ", graphite_vae_ba[i])
    print("\n")

    print("======================================")