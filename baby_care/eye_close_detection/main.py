import cv2
import numpy as np
import base64

def detect_eyes(base64_string):
    eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
    face_cascPath = 'haarcascade_frontalface_alt.xml'  #face detect model
    faceCascade = cv2.CascadeClassifier(face_cascPath)
    eyeCascade = cv2.CascadeClassifier(eye_cascPath)

    # Decode the base64 string
    img_data = base64.b64decode(base64_string)
    # Convert the data to a numpy array
    nparr = np.fromstring(img_data, np.uint8)
    # Convert the numpy array to an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
        frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
        eyes = eyeCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(eyes) == 0:
            return False  # Eyes are closed
        else:
            return True  # Eyes are open
    else:
        return None  # No face detected
