import os
import tensorflow
from os.path import join
from IPython.display import Image, display
from learntools.deep_learning.decode_predictions import decode_predictions
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#=============================================================================================================
#Choose Images to Work With
hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'
hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

img_paths = hot_dog_paths + not_hot_dog_paths

#=============================================================================================================
# Run an Example Model

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)

#=============================================================================================================
# Create Model with Pre-Trained Weights File. Make Predictions

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

most_likely_labels = decode_predictions(preds, top=3)


#=============================================================================================================
# Visualize Predictions

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])


#=============================================================================================================
# function that takes the models predictions (in the same format as `preds` from the set-up code) and 
# returns a list of `True` and `False` values
decoded = decode_predictions(preds, top=1)
print(decoded)

def is_hot_dog(preds):
    decoded = decode_predictions(preds, top=1)

    # pull out predicted label, which is in d[0][1] due to how decode_predictions structures results
    labels = [d[0][1] for d in decoded]
    out = [l == 'hotdog' for l in labels]
    return out


#=============================================================================================================
# Evaluate Model Accuracy
def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
    # We'll use the counts for denominator of accuracy calculation
    num_hot_dog_images = len(paths_to_hotdog_images)
    num_other_images = len(paths_to_other_images)
    
    hot_dog_test_data = read_and_prep_images(paths_to_hotdog_images)
    hot_dog_preds = model.predict(hot_dog_test_data)
    num_correct_hotdog_preds = sum(is_hot_dog(hot_dog_preds))
    
    other_images_test_data = read_and_prep_images(paths_to_other_images)
    other_images_preds = model.predict(other_images_test_data)
    num_correct_other_images = num_other_images - sum(is_hot_dog(other_images_preds))
    
    total_correct = num_correct_hotdog_preds + num_correct_other_images
    total_preds = num_hot_dog_images + num_other_images
    return total_correct / total_preds

# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code
my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)
print("Fraction correct in small test set: {}".format(my_model_accuracy))

