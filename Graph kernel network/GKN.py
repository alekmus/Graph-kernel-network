import tensorflow as tf
import tensorflow.keras as tfk
import spektral

class GKNet(tfk.models.Model):
    """[summary]
    """
    def __init__(self, 
                 channels, 
                 depth, 
                 kernel_layers, 
                 adj_matrix, 
                 edge_features,
                 n_out,
                 name = 'Graph_kernel_network',
                 **kwargs):
        
        super(GKNet, self).__init__(name=name, **kwargs)
        self.adj_matrix = adj_matrix
        self.edge_features = edge_features
        self.depth = depth

        self.conv_layers = []
        """
        self.conv_layers1 = spektral.layers.EdgeConditionedConv(channels, kernel_layers, activation=tf.nn.relu)
        self.conv_layers2 = spektral.layers.EdgeConditionedConv(channels, kernel_layers, activation=tf.nn.relu)
        """
        self.pool = spektral.layers.GlobalSumPool()
        self.dense = tfk.layers.Dense(n_out)
        
        for i in range(depth):
            self.conv_layers.append(spektral.layers.EdgeConditionedConv(channels, kernel_layers, activation=tf.nn.relu))
        

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        E = inputs[2]
        
        """
        X = self.conv_layers1([X, A, E])
        X = self.conv_layers2([X, A, E])
        X = self.pool(X)
        return self.dense(X)
        """
        for i in range(self.depth):
            X = self.conv_layers[i]([X, A, E])           
        return self.dense(self.pool(X))
       