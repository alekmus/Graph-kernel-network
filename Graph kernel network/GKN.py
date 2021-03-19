import tensorflow as tf
import tensorflow as tf
import tensorflow.keras as tfk
import spektral
import tensorflow.keras.backend as K

class GKNet(tfk.models.Model):
    def __init__(self, 
                 channels, 
                 depth, 
                 kernel_layers, 
                 name = 'Graph_kernel_network',
                 **kwargs):
        """
        Args:
            channels (int): Number of output features in edge convolution.
            depth (int): Number of message passing time steps.
            kernel_layers (list): A list of integers denoting number of neurons in each hidden layer
                                    of the multilayer perceptron used in the edge convolution.
                                    Example: [8,16,8] for three hidden layers with 8,16 and 8 neurons.
            name (str, optional): Name of the network. Defaults to 'Graph_kernel_network'.
        """
        
        super().__init__(name=name, **kwargs)
        self.depth = depth
        self.conv_layer = spektral.layers.ECCConv(channels, kernel_layers)
        self.norm_layers = [tfk.layers.LayerNormalization() for _ in range(depth)]     

        
    
    def call(self, input):
        X = input[0]
        A = input[1]
        E = input[2]

        for i in range(self.depth):
            X = self.conv_layer([X,A,E])
            X = tf.nn.leaky_relu(X)
            X = self.norm_layers[i](X)
        
        out = self.conv_layer([X,A,E])
        return tf.math.reduce_max(out,axis=1, keepdims=True)