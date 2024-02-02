from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model_path = "keras_model.h5"
model = load_model(model_path, compile=False)
