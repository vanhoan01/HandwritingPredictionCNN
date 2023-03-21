from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
model = tf.keras.models.load_model('./weights/')
app = FastAPI()
image_shape = (28, 28)
class Response(BaseModel):
 filename: str
 predict_number: int
 probability: float
@app.post('/mnist_prediction/', response_model=Response)
async def number_prediction(file: UploadFile = File(...)):
 contents = await file.read()
 jpg_as_np = np.frombuffer(contents, dtype=np.uint8)
 image = cv2.imdecode(jpg_as_np, 0)
 #Resize ảnh về kích cỡ yêu cầu của mô hình
 image = cv2.resize(image, image_shape, interpolation = cv2.INTER_AREA)
 #Reshape ảnh cho đúng chiều
 image = np.expand_dims(image, axis = 0)
 image = np.expand_dims(image, axis = 3)
 #Preprocessing dữ liệu về khoảng 0-1
 image = image/255.0
 output = model.predict(image)
 return {
 'filename': file.filename,
 'predict_number': output.argmax(),
 'probability': output[0][output.argmax()]
 }