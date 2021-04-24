import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model('models/22_04_2021.h5')


class Images(BaseModel):
    name: str
    img: List[int] = None


@app.get('/')
def index():
    return {'message': 'This is the EcoLoving server!'}


@app.post('/predict')
def predict_label(data: Images):
    """ FastAPI
    Args:
        data (Images): json file
    Returns:
        prediction: image label
    """
    res = {'battery': 0,
     'cans': 1,
     'glassbottle': 2,
     'medicinebottle': 3,
     'milkbox': 4,
     'pesticidebottle': 5,
     'plastic_cont_mark1': 6,
     'plasticbottle': 7,
     'toothbrush': 8,
     'toothpastetube': 9}
    res = list(res.keys())
    data = data.dict()
    img = data['img']
    img = np.array(img, dtype='uint8').reshape(1, 128, 128, 3)
    img_f = tf.cast(img, tf.float32)
    prediction = model.predict(img_f)
    return {
        'prediction': res[prediction.argmax()]
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5432)

