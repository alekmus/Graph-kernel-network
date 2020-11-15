import tensorflow as tf
import tensorflow.keras as tfk
import spektral


class GKNet(tf.Module):

    def __init__(self,
                 width,
                 ker_width,
                 depth,
                 ker_in,
                 in_width=1,
                 out_width=1,
                 name=None):
        """
        :param width:
        :param ker_width:
        :param depth:
        :param ker_in:
        :param in_width:
        :param out_width:
        :param name:
        """
        super(GKNet, self).__init__(name=name)
        self.depth = depth
        self.layers = []

        kernel = generate_mlp(ker_in,
                              ker_width,
                              ker_width,
                              width**2,
                              activation)
        self.GNN = spektral.layers.EdgeConditionedConv()
    def generate_mlp(ker_in,
                     ker_width,
                     ker_width,
                     width**2,
                     activation):
