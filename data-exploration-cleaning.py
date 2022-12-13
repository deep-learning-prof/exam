'''
This script downloads a weather data set, and displays some of its properties. 
'''


'''
Import TensorFlow and other required libraries
'''
import tensorflow as tf
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from windowgenerator import WindowGenerator as wg
'''
The weather dataset. This dataset contains 14 different features such as air temperature,
atmospheric pressure, and humidity. These were collected every 10 minutes, beginning in 2003.
For efficiency, you will use only a few years of the data. This section of the
dataset was prepared by François Chollet for his book Deep Learning with Python 
'''

'''
Download the dataset
'''

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


'''
Clean the data. We first simplify the data by only keeping hourly measurements. 
'''
df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

'''
We next delete erroneous velocity measurements
'''
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()

'''
Convert wind directions and magnitudes to wind vectors
'''
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)


'''
Data Visualization. Lets take a look at the temperature, peressure and humidity columns of our data
We first plot all data. 
'''
df_plot = df
df_plot.index = date_time

fig, axes = plt.subplots(nrows=3, ncols=1)
df_plot['T (degC)'].plot(ax=axes[0]); axes[0].set_title('Temperature (degC)')
df_plot['p (mbar)'].plot(ax=axes[1]); axes[1].set_title('Pressure (mbar)')
df_plot['rho (g/m**3)'].plot(ax=axes[2]); axes[2].set_title('Humidity (g/m**3)')
fig.subplots_adjust(hspace=2)
fig.savefig('sample-data-all.png')



'''
We next only plot the first 480 hours. 
'''
df_plot = df[:][:480]
df_plot.index = date_time[:480]

fig, axes = plt.subplots(nrows=3, ncols=1)
df_plot['T (degC)'].plot(ax=axes[0]); axes[0].set_title('Temperature (degC)')
df_plot['p (mbar)'].plot(ax=axes[1]); axes[1].set_title('Pressure (mbar)')
df_plot['rho (g/m**3)'].plot(ax=axes[2]); axes[2].set_title('Humidity (g/m**3)')
fig.subplots_adjust(hspace=2)
fig.savefig('sample-data-480.png')

'''
Data description
'''
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns',50)
print(df.describe())

'''
FInally, we convert dates to timestamps and add additional columns that capture the fact the weather
changes prediocally within a dat and within a year. 
'''
timestamp_s = date_time.map(pd.Timestamp.timestamp).to_numpy()
day = 24*60*60
year = (365.2425)*day



df['Day sin'] = np.sin(timestamp_s *(2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


'''
Split data into training and testing sets. We separate the first 70% of the data for training
and the rest for testing. We avoid random shuffle since this is sequential data. When we train
the model we will need windows of sequential data. 
'''

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
test_df = df[int(n*0.7):]

num_features = df.shape[1]


'''
Normalization. This is a routing step to ensure all data is within the same range, which helps with
numerical stability. 
'''
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

'''
Window generation test. We test the window generation here. 
'''
w1 = wg(input_width=24, label_width=1, shift=24, 
        train_df=train_df, test_df=test_df,
        label_columns=['T (degC)'],
        )

print(w1)

'''
We test the window split_window method here. 
'''

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])

example_inputs, example_labels = w1.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

'''
Window visualization. We also plot the three samples in the batch
'''
w1.example = example_inputs, example_labels
w1.plot(filename='temp-window.png')

w1.plot(plot_col='p (mbar)', filename='pressure-window.png')

'''
Save the dataset. 
'''
train_df.to_csv('train_data.csv')
test_df.to_csv('test_data.csv')



