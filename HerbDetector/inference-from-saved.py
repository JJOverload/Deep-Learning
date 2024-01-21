import numpy as np
import keras

image_size = (180, 180)

print("Loading model.")
model = keras.saving.load_model("final_model_herb.keras")

print("Running inference on new data.")
img = keras.utils.load_img(
	"herb_archive/rosemary-archive/rosemary-herb_1a2.jpeg",
	color_mode="grayscale",
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
