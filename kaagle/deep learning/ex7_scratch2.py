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

def prep_data(raw):
    y = raw[:, 0] 											# all rows, only first column
    out_y = keras.utils.to_categorical(y, num_classes)		
    
    x = raw[:,1:]											# all rows, second columns to end
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)

#=============================================================================================================
# Start the model
fashion_model = Sequential()

#=============================================================================================================
# Add the first layer - 12 filters, 3 kernel_size
fashion_model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

#=============================================================================================================
# Add the remaining layers
fashion_model.add(Conv2D(20, activation='relu', kernel_size=3))		 # Add 2 more convolutional (`Conv2D layers`) with 20 filters each, 'relu' activation, and a kernel size of 3.
fashion_model.add(Flatten())										                   # Follow that with a `Flatten` layer
fashion_model.add(Dense(100, activation='relu'))					         # A `Dense` layer with 100 neurons
fashion_model.add(Dense(10, activation='softmax'))					       # Add your prediction layer to `fashion_model`.  This is a `Dense` layer.  
																                                   # We alrady have a variable called `num_classes`.  Use this variable when specifying the number of nodes in this layer. 
																                                   # The activation should be `softmax`

#=============================================================================================================
# Compile Your Model
fashion_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#=============================================================================================================
# Fit The Model - The data used to fit the model. First comes the data holding the images, and second is the data with the class labels to be predicted. 
				# Look at the first code cell (which was supplied to you) where we called `prep_data` to find the variable names for these.
fashion_model.fit(x, y,
                  batch_size=128,
                  epochs=2,
                  validation_split = 0.2)

