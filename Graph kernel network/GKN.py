import tensorflow as tf
import tensorflow.keras as tfk
import spektral

class GKNet(tfk.models.Model):
    def __init__(self, 
                 channels, 
                 depth, 
                 kernel_layers, 
                 n_out,
                 name = 'Graph_kernel_network',
                 **kwargs):
        
        super(GKNet, self).__init__(name=name, **kwargs)
        self.depth = depth

        self.conv_layers = []
        self.dense = tfk.layers.Dense(n_out)
        self.pool = spektral.layers.GlobalSumPool()
        for i in range(depth):
            self.conv_layers.append(spektral.layers.EdgeConditionedConv(channels, kernel_layers, activation=tf.nn.relu))
        

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        E = inputs[2]
    
        for conv_layer in self.conv_layers:
            X = conv_layer([X, A, E])           

        return self.dense(self.pool(X))
       