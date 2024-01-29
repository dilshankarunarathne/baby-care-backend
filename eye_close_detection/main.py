import cv2
import numpy as np
import base64

eye_cascPath = 'eye_close_detection/haarcascade_eye_tree_eyeglasses.xml'
face_cascPath = 'eye_close_detection/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier("eye_close_detection/haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)


def detect_face_or_not(img):    # TODO bugfix
    """
        Detects whether a face is detected or not in a image.

        This function decodes the base64 string to get the image data, converts the data to a numpy array,
        and then uses OpenCV to detect whether a face is detected or not. If no face is detected in the image,
        the function returns False. If a face is detected, the function returns True.

        Args:
            img (str): string of the image.

        Returns:
            bool: True if eyes are open, False if eyes are closed, None if no face is detected.
    """

    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) > 0:
        return True  # Face is detected
    else:
        return False  # No face detected


def detect_eyes(img):
    """
        Detects whether eyes are open or closed in a base64 encoded image.

        This function decodes the base64 string to get the image data, converts the data to a numpy array,
        and then uses OpenCV to detect whether the eyes are open or closed. If no face is detected in the image,
        the function returns None. If a face is detected but no eyes are detected, the function returns False
        (indicating that the eyes are closed). If a face and eyes are detected, the function returns True
        (indicating that the eyes are open).

        Args:
            img (str): string of the image.

        Returns:
            bool: True if eyes are open, False if eyes are closed, None if no face is detected.
    """

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
        frame = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
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


def check_image_for_minors(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    print("Found {0} faces!".format(len(faces)))

    result_frame = frame.copy()
    if len(faces) != 0:  # If there are faces in the images
        for (x, y, w, h) in faces:  # For each face in the image
            # get the rectangle img around all the faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 5)
            sub_face = frame[y:y + h, x:x + w]

            # apply a gaussian blur on this new recangle image
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            # merge this blurry rectangle to our final image
            result_frame[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

    return len(faces) > 0  # Return True if a face is detected, False otherwise
