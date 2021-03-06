from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

#=============================================================================================================
## Specify Model

num_classes = 2
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#=============================================================================================================
# Start the model
my_new_model = Sequential()

#=============================================================================================================
# Add the remaining layers
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False # Say not to train first layer (ResNet) model. It is already trained

#=============================================================================================================
##Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#=============================================================================================================
## Fit Model
image_size = 224
# data augmentation
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        'train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=22,
        validation_data=validation_generator,
        validation_steps=1)
