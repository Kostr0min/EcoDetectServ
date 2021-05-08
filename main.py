import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import numpy as np

app = FastAPI()

class Images(BaseModel):
    name: str


@app.get('/')
def index():
    return {'message': 'This is the EcoLoving server!'}


@app.get('/predict')
def predict_label(data: Images):
    """ FastAPI
    Args:
        data (Images): json file
    Returns:
        prediction: image label
    """
    data = data.dict()
    return {
        'prediction': data['name']
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5432)

