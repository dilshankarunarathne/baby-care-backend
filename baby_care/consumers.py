import asyncio
import json
import base64
import cv2
import numpy as np
import re

from channels.generic.websocket import AsyncWebsocketConsumer

from channels.generic.http import AsyncConsumer
import json

from baby_care.eye_close_detection.main import detect_eyes
from baby_care.face_recognition.detector import recognize_faces


class PostConsumer(AsyncConsumer):
    async def websocket_connect(self, event):
        await self.send({
            'type': 'websocket.accept'
        })

    async def websocket_receive(self, event):
        data = json.loads(event['text'])
        # process data
        # return response
        await self.send({
            'type': 'websocket.send',
            'text': json.dumps({'message': 'Data received'})
        })


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        # Extract base64 string from data URL
        base64_str = re.search(r'base64,(.*)', text_data).group(1)
        frame_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("error: frame is none")
        else:
            print("Image decoded successfully")

        # face recognition
        recognized_name = recognize_faces(base64_str)
        print("Face identified as: ", recognized_name)

        # eye open/close detection
        eyes = detect_eyes(base64_str)
        if eyes is None:
            etxt = "No face detected"
        elif eyes:
            etxt = "Eyes open"
        else:
            etxt = "Eyes closed"
        print(etxt)

        # pose estimation
        pose = None

        await self.send(text_data=json.dumps({"recognized_name": recognized_name, "eyes": etxt, "pose": pose}))


def check_base64_image(base64_string, output_file):
    try:
        with open(output_file, 'wb') as f_out:
            f_out.write(base64.b64decode(base64_string))
        print(f"Image written to {output_file}. Please check if it's a valid image.")
    except Exception as e:
        print(f"Failed to decode and write image: {e}")
