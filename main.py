from fastapi import FastAPI

from routes import auth, image_route

app = FastAPI()

app.include_router(auth.router)
app.include_router(image_route.router)
