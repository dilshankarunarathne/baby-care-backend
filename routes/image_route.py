from fastapi import APIRouter, UploadFile, File, Depends

from auth.authorize import oauth2_scheme, get_current_user, credentials_exception

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

    # check if image is a baby

    # check if baby is asleep
    detect_eyes(encoded_image)

    # estimate sleep position
