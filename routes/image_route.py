import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends

from auth.authorize import oauth2_scheme, get_current_user, credentials_exception
from eye_close_detection.main import detect_eyes
from face_rec.detector import recognize_faces

router = APIRouter(
    prefix="/api/image",
    tags=["image"],
    responses={404: {"description": "The requested url was not found"}},
)


@router.post("/verify")
async def verify_baby_image_endpoint(
        image: UploadFile = File(...),
        token: str = Depends(oauth2_scheme)
):
    user = await get_current_user(token)

    if user is None:
        raise credentials_exception

    # decode image data
    if image and image.content_type != "image/jpeg":
        return {300: {"description": "Only jpeg images are supported"}}
    else:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("error frame is none")
        else:
            print("Image decoded successfully")

    # detect face
    faces = recognize_faces(frame)

    return {"faces": faces}


@router.post("/check")
async def check_baby_image_endpoint(
        image: UploadFile = File(...),
        token: str = Depends(oauth2_scheme)
):
    user = await get_current_user(token)

    if user is None:
        raise credentials_exception

    # decode image data
    if image and image.content_type != "image/jpeg":
        return {300: {"description": "Only jpeg images are supported"}}
    else:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("error frame is none")
        else:
            print("Image decoded successfully")

    # check if image is a baby

    # check if baby is asleep
    eyes = detect_eyes(frame)
    eye_text = _get_eye_text(eyes)

    # estimate sleep position

    return {"eyes": eye_text}


def _get_eye_text(eye_data):
    # True if eyes are open, False if eyes are closed, None if no face is detected.
    if eye_data is None:
        return "No face detected"
    elif eye_data:
        return "Eyes are open"
    else:
        return "Eyes are closed"
