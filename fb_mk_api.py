import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import joblib
import os
import torch
from prepare_image_data import PrepareImageData
from transforms import TransformImage, TransformText
from PIL import Image
import nltk
from image_model import ImageModel
from text_model import TextModel
from combined_model import CombinedModel
import torch.nn as nn
from os.path import join

# IMAGE PREDICTION
# load the saved pytorch transformer used for training
saved_transformer = joblib.load(os.path.join(os.getcwd(), 'models', 'img_transformer.pkl'))
# get the preprocessing function used for training
img_preprocessor = PrepareImageData.process_image
# instantiate the final transformer object
transformer_img = TransformImage(saved_transformer, img_preprocessor)
# path for the saved cnn model 
saved_img_model_path = os.path.join(
    os.getcwd(), 'models', 'best_image_cnn_model.pt')
# path for the saved class decoder
decoder = joblib.load(os.path.join(os.getcwd(), 'models', 'cat_decoder.pkl'))
# instantiate the api model
imageModel = ImageModel(saved_img_model_path, decoder, transformer_img)
imageModel.to(torch.device("cpu"))
imageModel.eval()


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
    uvicorn.run('fb_mk_api:api', port=8008, host='0.0.0.0')

    

