import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

#=============================================================================================================
## Specify Model
img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_size = 30000
train_file = "../input/digit-recognizer/train.csv"
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)

#=============================================================================================================
# Start the model
batch_size = 16

fashion_model = Sequential()

#=============================================================================================================
# Add the first layer - 16 filters, 3 kernel_size
fashion_model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

#=============================================================================================================
# Add the remaining layers
fashion_model.add(Conv2D(16, (3, 3), activation='relu'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))

#=============================================================================================================
# Compile Your Model
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#=============================================================================================================
# Fit The Model - The data used to fit the model. First comes the data holding the images, and second is the data with the class labels to be predicted. 
        # Look at the first code cell (which was supplied to you) where we called `prep_data` to find the variable names for these.
fashion_model.fit(x, y,
          batch_size=batch_size,
          epochs=3,
          validation_split = 0.2)