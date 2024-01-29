from fastapi import APIRouter, UploadFile, File, Depends

router = APIRouter(
    prefix="/api/image",
    tags=["image"],
    responses={404: {"description": "The requested url was not found"}},
)


@router.post("/verify")
async def verify_baby_image_endpoint(
        image: UploadFile = File(...),
        token: str = Depends(oauth2_scheme)
)
