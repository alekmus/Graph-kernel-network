from pickle import load

from tensorflow.python.keras.backend import dtype
import gkn
from graph import EIT_dataset
import tensorflow as tf
import tensorflow.keras as tfk
import utilities, datetime



if __name__== '__main__':
    BATCH_SIZE = 1
    EPOCHS = 20
    
    # Load data and convert .mat files if necessary
    data = EIT_dataset('mat_data','graph_data')
    # Define loader to create minibatches
    loader = utilities.WDJLoader(data, 
                                 node_level=True, 
                                 batch_size = BATCH_SIZE)

    model = gkn.generate_EITNet()

    model.fit(loader.load(), 
              epochs=EPOCHS,
              steps_per_epoch=loader.steps_per_epoch)

    model.save_weights(f'weights/EITNet_weights_{datetime.datetime.now().strftime("%d%m%Y_%H%M%S")}')