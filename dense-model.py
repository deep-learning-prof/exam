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

'''
Load the data
'''
train_df = pd.read_csv('train_data.csv', index_col='Date Time')
test_df = pd.read_csv('test_data.csv', index_col = 'Date Time')

'''
One-hour input dense model
'''
'We first design a one-step window generator'
window = wg(
            input_width=1, 
            label_width=1, 
            shift=1,
            train_df=train_df, 
            test_df=test_df,
            label_columns=['T (degC)']
            )

'''
We then build a linear deep feedforward network. If we do not define a activaion function, the
defauls is linear.
'''
dense = tf.keras.Sequential()
dense.add(tf.keras.layers.Dense(units=64, activation='relu'))
dense.add(tf.keras.layers.Dense(units=64, activation='relu'))
dense.add(tf.keras.layers.Dense(units=1))


dense.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = dense.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.test,)

'Lets visualize the model'
tf.keras.utils.plot_model(dense, to_file="dense_model.png", show_shapes=True)

'Save the training history'
history_df = pd.DataFrame(history.history)
history_df.to_csv('dense-model-training-history.csv')

'Save the model'
dense.save('trained-dense-model')
