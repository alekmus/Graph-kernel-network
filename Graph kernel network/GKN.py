import tensorflow as tf
import tensorflow.keras as tfk
import spektral


class GKNet(tfk.Model):
    """[summary]
    """
    def __init__(self, channels, depth, kernel_layers, activation=tf.nn.relu):
        """[summary]

        Args:
            channels ([type]): [description]
            depth ([type]): [description]
            kernel_layers ([type]): [description]
            activation ([type], optional): [description]. Defaults to tf.nn.relu.

        Returns:
            [type]: [description]
        """

        super(GKNet, self).__init__()
        self.depth = depth
        self.activation = activation
        self.conv = spektral.layers.EdgeConditionedConv(channels, [kernel_layers], activation=tf.nn.relu)
      


        def call(self, inputs):
            node_features, adjacency_matrix, edge_features = inputs
            for i in range(self.depth):
                node_features = self.activation(self.conv([node_features, adjacency_matrix, edge_features]))
            
            return node_features
