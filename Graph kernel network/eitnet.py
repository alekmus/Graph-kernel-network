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
    width = 512
    model = gkn.GKNet(64, 6, [width, width])
    optimizer = tfk.optimizers.RMSprop(learning_rate=0.0000001, centered=True, momentum=0.0)
    model.compile(optimizer, loss='mse', metrics=[masked_MAPE])
    return model

if __name__== '__main__':
    BATCH_SIZE = 1
    EPOCHS = 300
    # Load data and convert .mat files if necessary
    data = EIT_dataset('mat_data')
    # Inplace operation
    np.random.shuffle(data)
    split_i = int(data.n_graphs*0.1)
    val_data = data[:split_i]
    train_data = data[split_i:]
    # Define loader to create minibatches
    loader = spektral.data.loaders.SingleLoader(data[:1])
    val_loader = utilities.WDJLoader(val_data[:1], batch_size = BATCH_SIZE, node_level=True)
   
    model = generate_EITNet()
    model.load_weights("weights/norm_eit_checkp")                               
    history = model.fit(loader.load(), 
              epochs=EPOCHS,
              steps_per_epoch=loader.steps_per_epoch,
              callbacks=[tfk.callbacks.ModelCheckpoint("weights/norm_eit_checkp", save_freq=EPOCHS)])

    for _ in range(2):
        for i in range(1,train_data.n_graphs):
            loader = spektral.data.loaders.SingleLoader(data[i:i+1])
            #model.load_weights("weights/norm_eit_checkp")                
            history = model.fit(loader.load(), 
                    epochs=EPOCHS,
                    steps_per_epoch=loader.steps_per_epoch,
                    callbacks=[tfk.callbacks.ModelCheckpoint("weights/norm_eit_checkp", save_freq=EPOCHS)])
        np.random.shuffle(data)

    #print(model.summary())
    model.save_weights(f'weights/norm_EITNet_weights_{datetime.datetime.now().strftime("%d%m%y")}', overwrite=True)
    model.evaluate(val_loader.load(), steps=val_loader.steps_per_epoch)
