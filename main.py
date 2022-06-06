import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf

app = FastAPI()


CLASS_NAMES = ["coco", "docks", "notdocks"]
model1 = tf.keras.models.load_model("../saved_models/2")


@app.get("/ping")
async def ping():
    return"hello ,i am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)

):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)
    predictions = model1.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]*100)
    return{
        'class ': predicted_class,
        'confidence ': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8800")
