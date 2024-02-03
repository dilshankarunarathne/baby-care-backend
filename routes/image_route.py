import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends

from auth.authorize import oauth2_scheme, get_current_user, credentials_exception
from eye_close_detection.main import detect_eyes, detect_face_or_not, check_image_for_minors
from face_rec.detector import recognize_faces
from face_rec.tm_main import analyze_image
from posture.main import analyze_posture

router = APIRouter(
    prefix="/api/image",
    tags=["image"],
    responses={404: {"description": "The requested url was not found"}},
)

last_name = "Unknown"


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

    # analyze image
    class_name, confidence_score = analyze_image(frame)

    return {"name": class_name, "confidence_score": confidence_score}


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
    face = detect_face_or_not(frame)
    # face_text = _get_face_text(face)

    # check if baby is asleep
    eyes = detect_eyes(frame)
    eye_text = _get_eye_text(eyes)

    # baby detection trial 2
    # baby_det = check_image_for_minors(frame)

    class_name, confidence_score = analyze_image(frame)

    # estimate sleep position
    pose, pose_confidence = analyze_posture(frame)

    # return data
    return {"sleep": eye_text, "pose": pose}


def _get_face_text(face_data):
    # True if face is detected, False if no face is detected.
    if face_data is None:
        return "No face detected"
    elif face_data:
        return "Face detected"
    else:
        return "No face detected"


def _get_eye_text(eye_data):
    # True if eyes are open, False if eyes are closed, None if no face is detected.
    if eye_data is None:
        return "Baby is asleep"
    elif eye_data:
        return "Baby is awake"
    else:
        return "Baby is asleep"
