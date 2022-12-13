"""
This file is based on the TensorFlow tutorials
"""

"""
### Import TensorFlow
"""
import matplotlib.pyplot as plt
import pandas as pd #we will use this to save the training log
import tensorflow as tf
from windowgenerator import WindowGenerator as wg
"""
We first load the training log
"""
modelName ='cnn-model' 
history = pd.read_csv(modelName + '-training-history.csv')


"""
### We can see the training accuracy and testing accuracy as follows. 
"""
plt.plot(history['mean_absolute_error'], label='mean_absolute_error')
plt.plot(history['val_mean_absolute_error'], label = 'val_mean_absolute_error')
plt.xlabel('Epoch')
plt.ylabel('mean_absolute_error')
plt.ylim([0, .08])
plt.legend(loc='upper right')
plt.savefig(modelName+'-mae-plot.png')
plt.clf()


"""
### We can see the training and testing error as follows. 
"""
plt.plot(history['loss'], label='Training Error (Mean squared error)')
plt.plot(history['val_loss'], label = 'Testing Error (Mean squared error)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, .08])
plt.legend(loc='upper right')
plt.savefig(modelName+'-training-error-plot.png')

'''
Visualize Prediction. We can feed a window to the model to see a few sample predictions. 
'''

train_df = pd.read_csv('train_data.csv', index_col='Date Time')
test_df = pd.read_csv('test_data.csv', index_col = 'Date Time')


wide_window = wg(
    input_width=24, label_width=24, shift=1, train_df=train_df, test_df=test_df,
    label_columns=['T (degC)'])

#example_inputs, example_labels = wide_window.split_window


#linear = tf.keras.models.load_model('trained-linear-model')
#wide_window.plot(model=linear)

