from pickle import load
import gkn
from graph import EIT_dataset
import tensorflow as tf
import tensorflow.keras as tfk
import utilities
# Needs to be imported eventhough "not used"
# because pickle requires the pickled class
# to be explicitly defined within the 
# namespace of this module.
from graph import mat_graph

if __name__=='__main__':
    BATCH_SIZE = 32
    EPOCHS = 1
    
    # Load data and convert .mat files if necessary
    data = EIT_dataset('mat_data')
    # Define loader to create minibatches

    

    loader = utilities.WDJLoader(data, 
                                 node_level=True, 
                                 batch_size = BATCH_SIZE)
   
    model = gkn.GKNet(32,5,[16,16,16])
    model.compile('adam', 'mse')
    model.fit(loader.load(), 
              epochs=EPOCHS,
              steps_per_epoch=loader.steps_per_epoch)