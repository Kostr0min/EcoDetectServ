import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List

import numpy as np
import torch.nn as nn
from PIL import Image

import torchvision
from torchvision import datasets, models, transforms

import torch

app = FastAPI()

model = torchvision.models.mobilenet_v2(pretrained=True, progress=True)

model_ft = model
model_ft.fc = nn.Linear(1280, 6)

model_ft.load_state_dict(torch.load('./models/model_pytorch'))
model_ft.eval()


class Images(BaseModel):
    name: str
    img: List[int] = None
    shape: tuple


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
    class_names = ['cans',
     'glassbottle',
     'medicinebottle',
     'pesticidebottle',
     'plastic_cont_mark1',
     'plasticbottle']

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data = data.dict()
    img = data['img']
    shape = data['shape']
    img = np.array(img, dtype='uint8').reshape(shape[0], shape[1], 3)
    input_image = Image.fromarray(img)
    input_tensor = data_transforms['test'](input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model_ft.to('cuda')
    with torch.no_grad():
        output = model_ft(input_batch)
        _, preds = torch.max(output, 1)
        print(preds)

    return {
        'prediction': class_names[preds]
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5432)

