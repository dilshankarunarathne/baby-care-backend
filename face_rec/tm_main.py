from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("face_rec/keras_Model.h5", compile=False)

# Load the labels
class_names = open("face_rec/labels.txt", "r").readlines()


def analyze_image(img_data):
    # Convert the numpy array back to an image
    image = Image.fromarray(img_data)

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

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # strip() is used to remove any leading or trailing white spaces
    confidence_score = float(prediction[0][index])  # convert numpy.float32 to native Python float

    # return prediction and confidence score
    return class_name, confidence_score
