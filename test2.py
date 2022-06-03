from gae_train import gae_train
from graphite_train import graphite_train

t1 = graphite_train(epochs=100, dataset="erdos_renyi", learning_rate=0.01, vae=0)
t1 = graphite_train(epochs=100, dataset="erdos_renyi", learning_rate=0.01, vae=1)