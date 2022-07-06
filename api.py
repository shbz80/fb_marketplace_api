import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from prepare_image_data import PrepareImageData
from transforms import TransformImage
from PIL import Image

class ImageModel(torch.nn.Module):
    def __init__(self, saved_model_path, decoder, transformer) -> None:
        super().__init__()
        self.img_model = models.resnet50(pretrained=True)
        num_ftrs = self.img_model.fc.in_features
        self.img_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                            nn.ReLU(),
                            nn.Linear(256, len(decoder)),
                            )
        self.img_model.load_state_dict(torch.load(saved_model_path))
        self.decoder = decoder
        self.transformer = transformer

    def forward(self, input):
        input = self.transformer.transform(input)
        return self.img_model(input.unsqueeze(0))

    def predict(self, input):
        i = np.argmax(self.predict_proba(input)).item()
        return self.decoder[i]

    def predict_proba(self, input):
        output = self.forward(input).squeeze()
        return F.softmax(output, dim=0).detach()

    def predict_classes(self, input):
        proba = self.predict_proba(input)
        proba_args = proba.argsort().numpy()[::-1]
        classes = [self.decoder[arg] for arg in list(proba_args)]
        return classes

# load the saved pytorch transformer used for training
saved_transformer = joblib.load(os.path.join(os.getcwd(), 'models', 'img_transformer.pkl'))

# get the preprocessing function used for training
img_preprocessor = PrepareImageData.process_image

# instantiate the final transformer object
transformer = TransformImage(saved_transformer, img_preprocessor)

# path for the saved cnn model 
model_path = os.path.join(os.getcwd(), 'models', 'best_image_cnn_model.pt')

# path for the saved class decoder
decoder = joblib.load(os.path.join(os.getcwd(), 'models', 'cat_decoder.pkl'))

# instantiate the api model
imageModel = ImageModel(model_path, decoder, transformer)
imageModel.to(torch.device("cpu"))

api = FastAPI()

@api.post('/image')
def predict_img_class(img: UploadFile = File(...)):
    img = Image.open(img.file)
    classes = imageModel.predict_classes(img)
    pred = imageModel.predict(img)
    res = JSONResponse(status_code=200, content={'pred': pred, 'classes': classes})
    return res

@api.post('/text')
def predict_txt_class(txt: UploadFile = Form(...)):
    res = JSONResponse(status_code=200, content={
                       'pred': 'NA', 'classes': 'NA'})
    return res

@api.post('/combined')
def predict_comb_class(img: UploadFile = File(...), txt: UploadFile = Form(...)):
    res = JSONResponse(status_code=200, content={
                       'pred': 'NA', 'classes': 'NA'})
    return res

if __name__ == '__main__':
    uvicorn.run(api, port=8008, host='127.0.0.1')

    

