from gae_train import gae_train
from graphite_train import graphite_train

seed = 21
dataset = "ego_net"

gae_er = gae_train(epochs=20, model="gcn_ae", dataset=dataset, seed=seed)
vgae_er = gae_train(epochs=20, model="gcn_vae", dataset=dataset, seed=seed)
graphite_ae_er = graphite_train(epochs=20, vae=0, dataset=dataset, seed=seed)
graphite_vae_er = graphite_train(epochs=20, vae=1, dataset=dataset, seed=seed)

print("GAE: ", gae_er)
print("Graphite-AE: ", graphite_ae_er)
print("VGAE: ", vgae_er)
print("Graphite-VAE: ", graphite_vae_er)
print("\n")