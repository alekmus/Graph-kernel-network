import tensorflow as tf
import tensorflow.keras as tfk
import spektral

class GKNet(tfk.models.Model):
    def __init__(self, 
                 channels, 
                 depth, 
                 kernel_layers, 
                 n_labels = 1,
                 name = 'Graph_kernel_network',
                 **kwargs):
        """
        Args:
            channels (int): Number of output layers in edge convolution.
            depth (int): Number of message passing time steps.
            kernel_layers (list): A list of integers denoting number of neurons in each hidden layer
                                    of the multilayer perceptron used in the edge convolution.
                                    Example: [8,16,8] for three hidden layers with 8,16 and 8 neurons.
            name (str, optional): Name of the network. Defaults to 'Graph_kernel_network'.
        """
        
        super().__init__(name=name, **kwargs)
        self.depth = depth
        self.conv_layers = []

        for _ in range(depth-1):
            self.conv_layers.append(spektral.layers.ECCConv(channels, kernel_layers, activation=tf.nn.relu))

        self.output_layer = spektral.layers.ECCConv(n_labels, kernel_layers)
        self.conv_layers.append(self.output_layer)
    
        """
        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            trainable_params = self.trainable_variables
            grads = tape.gradient(loss, trainable_params)
            self.optimizer.apply_gradients(zip(grads, trainable_params))
            self.compiled_metrics.update_state(y,y_pred)

            return {m.name: m.result() for m in self.metrics}
        """
    def call(self, input):
        X = input[0]
        A = input[1]
        E = input[2]

        for conv in self.conv_layers:
            X = conv([X,A,E])
        return X
       