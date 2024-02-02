from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model_path = "keras_model.h5"
model = load_model(model_path, compile=False)

# Load the labels
labels_path = "labels.txt"
class_names = open(labels_path, "r").readlines()


def analyze_posture(img_data):
    # Replace this with the path to your image in Google Drive
    image_path = "test.jpg"
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the Keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:])
    print("Confidence Score:", confidence_score)
