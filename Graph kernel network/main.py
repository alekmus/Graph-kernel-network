import gkn
from graph import EIT_dataset
import datetime, utilities
import tensorflow.keras as tfk
import spektral

if __name__== '__main__':
    BATCH_SIZE = 1
    EPOCHS = 1
    
    # Load data and convert .mat files if necessary
    data = EIT_dataset('mat_data')
    # Define loader to create minibatches
    loader = spektral.data.loaders.DisjointLoader(data, batch_size = BATCH_SIZE)

    model = gkn.generate_EITNet()

    model.fit(loader.load(), 
              epochs=EPOCHS,
              steps_per_epoch=loader.steps_per_epoch,
              callbacks=[tfk.callbacks.ModelCheckpoint("weights/checkp",monitor='loss',save_best_only=True, save_freq=50)])

    model.save_weights(f'weights/EITNet_weights_{datetime.datetime.now().strftime("%d%m%Y")}')