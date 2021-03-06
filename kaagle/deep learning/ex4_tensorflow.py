from os.path import join
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50
##from learntools.deep_learning.decode_predictions import decode_predictions
from IPython.display import Image, display
#=============================================================================================================
#Choose Images to Work With

image_dir = '/train/'

img_paths = [join(image_dir, filename) for filename in 
                           ['0246f44bb123ce3f91c939861eb97fb7.jpg',
                            '84728e78632c0910a69d33f82e62638c.jpg',
                            '8825e914555803f4c67b26593c9d5aff.jpg',
                            '91a5e8db15bccfb6cfa2df5e8b95ec03.jpg']]
#=============================================================================================================
# Function to Read and Prep Images for Modeling

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)
    
#=============================================================================================================
# Create Model with Pre-Trained Weights File. Make Predictions

my_model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)

#=============================================================================================================
# Visualize Predictions

most_likely_labels = decode_predictions(preds, top=3, class_list_path='imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])



    
