import base64
import io
import pickle
from collections import Counter
from pathlib import Path

import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


def recognize_faces(
        base64_image_data, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> list:
    """Recognizes faces in a base64-encoded image string and returns a list of detected names."""

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    image_data = base64.b64decode(base64_image_data)
    input_image = face_recognition.load_image_file(io.BytesIO(image_data))

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    names = []
    for unknown_encoding in input_face_encodings:
        name = _recognize_face(unknown_encoding, loaded_encodings)
        names.append(name if name else "Unknown")

    return names


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
