from graphite_layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from graphite_layers import *
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelVAE(Model):
    '''VGAE Model for reconstructing graph edges from node representations.'''
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, nhidden1, nhidden2, vae, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_label = placeholders['adj_orig']
        self.weight_norm = 0
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2
        self.vae = vae
        self.build()

    def encoder(self, inputs):

        hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.nhidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)(inputs)

        self.z_mean = GraphConvolution(input_dim=self.nhidden1,
                                       output_dim=self.nhidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(hidden1)

        self.z_log_std = GraphConvolution(input_dim=self.nhidden1,
                                          output_dim=self.nhidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(hidden1)

    def get_z(self, random):

        z = self.z_mean + tf.random_normal([self.n_samples, self.nhidden2]) * tf.exp(self.z_log_std)
        if not random or not self.vae:
          z = self.z_mean

        return z

    def make_decoder(self):
      return

    def decoder(self, z):

        reconstructions = InnerProductDecoder(input_dim=self.nhidden2,
                                      act=lambda x: x,
                                      dropout=0.,
                                      logging=self.logging)(z)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

    def _build(self):
  
        self.encoder(self.inputs)
        self.make_decoder()
        z = self.get_z(random = True)
        z_noiseless = self.get_z(random = False)
        if not self.vae:
          z = z_noiseless

        self.reconstructions = self.decoder(z)
        self.reconstructions_noiseless = self.decoder(z_noiseless)