from fastapi import APIRouter, UploadFile, File, Depends

from auth.authorize import oauth2_scheme, get_current_user, credentials_exception
from eye_close_detection.main import detect_eyes

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

    pass


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
        nparray = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
        
    # check if image is a baby

    # check if baby is asleep
    detect_eyes(encoded_image)

    # estimate sleep position
