# -*- coding: utf-8 -*-

import GKN as gkn
from graph import EIT_dataset
import datetime
import tensorflow.keras as tfk
import spektral, utilities

def generate_EITNet():
    model = gkn.GKNet(128, 5, [64, 128, 64])
    optimizer = tfk.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    model.compile(optimizer, loss='mse', metrics=['MAPE'])
    return model

if __name__== '__main__':
    BATCH_SIZE = 1
    EPOCHS = 10
    
    # Load data and convert .mat files if necessary
    val_data = EIT_dataset('val_mat_data')
    data = EIT_dataset('mat_data')
    
    # Define loader to create minibatches
    loader = utilities.WDJLoader(data, batch_size = BATCH_SIZE,node_level=True)
    val_loader = utilities.WDJLoader(val_data, batch_size = BATCH_SIZE,node_level=True)
    model = generate_EITNet()
    
    model.load_weights("weights/eit_checkp")                
    history = model.fit(loader.load(), 
              epochs=EPOCHS,
              validation_data=val_loader.load(),
              validation_batch_size=BATCH_SIZE,
              validation_steps=val_loader.steps_per_epoch,
              steps_per_epoch=loader.steps_per_epoch,
              callbacks=[tfk.callbacks.ModelCheckpoint("weights/eit_checkp",monitor='val_loss',save_best_only=True)])
    print(model.summary())
    model.save_weights(f'weights/EITNet_weights_{datetime.datetime.now().strftime("%d%m%y")}', overwrite=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(20,20))
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('convergence.png')

    model.evaluate(val_loader.load(), steps=val_loader.steps_per_epoch)
