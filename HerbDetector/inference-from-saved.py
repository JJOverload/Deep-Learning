import numpy as np
import keras

image_size = (180, 180)
#model_name = "final_model_herb.keras"
model_name = "save_at_37.keras"
image_path = "herb_archive/sage-archive/sage-herb_163.jpeg"

print("Loading model.")
model = keras.saving.load_model(model_name)

print("Running inference on new data.")
img = keras.utils.load_img(
	image_path,
	color_mode="rgb",
	target_size=image_size,
)


img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
print(predictions)

indexOfMax = np.argmax(predictions[0])

print("indexOfMax:", indexOfMax)

#score = float(keras.ops.sigmoid(predictions[0][0]))
#print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


print("Done")
