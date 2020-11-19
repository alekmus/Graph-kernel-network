import tensorflow as tf
import tensorflow.keras as tfk
import spektral

class GKNet(tfk.models.Model):
    def __init__(self, 
                 channels, 
                 depth, 
                 kernel_layers, 
                 name = 'Graph_kernel_network',
                 **kwargs):
        """[summary]

        Args:
            channels (int): Number of output layers in edge convolution.
            depth (int): Number of message passing time steps.
            kernel_layers (list): A list of integers denoting number of neurons in each hidden layer
                                    of the multilayer perceptron used in the endge convolution.
                                    Example: [8,16,8] for three hidden layers with 8,16 and 8 neurons.
            name (str, optional): Name of the network. Defaults to 'Graph_kernel_network'.
        """
        
        super(GKNet, self).__init__(name=name, **kwargs)
        self.depth = depth

        self.conv_layers = []

        for _ in range(depth):
            self.conv_layers.append(spektral.layers.EdgeConditionedConv(channels, kernel_layers, activation=tf.nn.relu))
        

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]
        E = inputs[2]
    
        for conv_layer in self.conv_layers:
            X = conv_layer([X, A, E])           

        return X
       