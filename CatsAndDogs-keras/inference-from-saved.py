#import os
#import numpy as np
import keras
#from keras import layers
#import tensorflow as tf
#from tensorflow import data as tf_data

image_size = (180, 180)


print("Loading model.")
model = keras.saving.load_model("final_model_cats_dogs.keras")
"""
We get to >90% validation accuracy after training for 25 epochs on the full dataset
(in practice, you can train for 50+ epochs before validation performance starts degrading).
"""

"""
## Run inference on new data

Note that data augmentation and dropout are inactive at inference time.
"""



print("Running inference on new data.")
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)



img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
