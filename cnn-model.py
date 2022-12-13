'''
This script build the weather prediction models
'''
'''
Import libraries
'''
import tensorflow as tf
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from windowgenerator import WindowGenerator as wg

'''
Set Constants
'''
MAX_EPOCHS = 20
CONV_WIDTH = 3
'''
Load the data
'''
train_df = pd.read_csv('train_data.csv', index_col='Date Time')
test_df = pd.read_csv('test_data.csv', index_col = 'Date Time')

'''
Three-hour input dense model
'''
'We first design a one-step window generator'
window = wg(
            input_width=3, 
            label_width=1, 
            shift=1,
            train_df=train_df, 
            test_df=test_df,
            label_columns=['T (degC)']
            )

#'''
#We then build a linear deep feedforward network. If we do not define a activaion function, the
#defauls is linear.
#'''
cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),) 
cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1))


cnn.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = cnn.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.test,)

'Lets visualize the model'
tf.keras.utils.plot_model(cnn, to_file="cnn_model.png", show_shapes=True)

'Save the training history'
history_df = pd.DataFrame(history.history)
history_df.to_csv('cnn-model-training-history.csv')

'Save the model'
cnn.save('trained-cnn-model')
