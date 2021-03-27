# -*- coding: utf-8 -*-

from tensorflow.python.distribute import tpu_strategy
import GKN as gkn
from graph import EIT_dataset
import datetime
import tensorflow.keras as tfk
import tensorflow as tf
import spektral, utilities
import tensorflow.keras.backend as K
import numpy as np

def mask_zero_preds(y_true, y_pred):
    zero_mask = K.equal(y_true, 0)
    zero_mask = K.cast(zero_mask, dtype=K.floatx())
    zero_mask = 1 - zero_mask
    
    y_true = y_true * zero_mask
    y_pred = y_pred * zero_mask

    return y_true, y_pred

def masked_mse(y_true, y_pred):
    masked_y_true, masked_y_pred = mask_zero_preds(y_true, y_pred)
    return tfk.losses.mean_squared_error(masked_y_true, masked_y_pred)

def masked_MAPE(y_true, y_pred):
    masked_y_true, masked_y_pred = mask_zero_preds(y_true, y_pred)
    return tfk.losses.mean_absolute_percentage_error(masked_y_true, masked_y_pred)


def generate_EITNet():
    width = 1024
    model = gkn.GKNet(64, 7, [width, width])
    optimizer = tfk.optimizers.RMSprop(learning_rate=0.00001, centered=True, momentum=0.8)
    model.compile(optimizer, loss=masked_mse, metrics=[masked_MAPE])
    return model

if __name__== '__main__':
    BATCH_SIZE = 1
    EPOCHS = 10000
    # Load data and convert .mat files if necessary
    data = EIT_dataset('mat_data')
    # Inplace operation
    np.random.shuffle(data)
    split_i = int(data.n_graphs*0.1)
    val_data = data[:split_i]
    train_data = data[split_i:]
    # Define loader to create minibatches
    loader = utilities.WDJLoader(train_data, batch_size = BATCH_SIZE,node_level=True)
    val_loader = utilities.WDJLoader(val_data, batch_size = BATCH_SIZE,node_level=True)
   
    model = generate_EITNet()
    
    model.load_weights("weights/norm_eit_checkp")                
    history = model.fit(loader.load(), 
              epochs=EPOCHS,
              validation_data=val_loader.load(),
              validation_batch_size=BATCH_SIZE,
              validation_steps=val_loader.steps_per_epoch,
              steps_per_epoch=loader.steps_per_epoch,
              callbacks=[tfk.callbacks.ModelCheckpoint("weights/norm_eit_checkp",save_freq=200)])
    #print(model.summary())
    model.save_weights(f'weights/norm_EITNet_weights_{datetime.datetime.now().strftime("%d%m%y")}', overwrite=True)

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
