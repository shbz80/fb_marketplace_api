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

# TEXT PREDICTION
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

EMBED_DIM = 128
# word_to_idx = joblib.load('/models/word_to_idx.pkl')
path = join(os.getcwd(), 'models', 'word_to_idx.pkl')
word_to_idx = joblib.load(path)
embeddings = nn.Embedding(len(word_to_idx), EMBED_DIM)
embeddings.load_state_dict(torch.load(join(os.getcwd(),
    'models','embeddings_wts.pt'), map_location=torch.device('cpu')))
embeddings.requires_grad_(False)
max_word_len = joblib.load(join(os.getcwd(), 'models', 'max_desc_len.pkl'))

transformer_txt = TransformText(
    word_to_idx, embeddings, max_word_len, EMBED_DIM)

saved_txt_model_path = os.path.join(
    os.getcwd(), 'models', 'best_text_cnn_model.pt')
textModel = TextModel(saved_txt_model_path, word_embd_dim=EMBED_DIM,
                      decoder=decoder, word_kernel_size=3)
textModel.to(torch.device("cpu"))
textModel.eval()

# COMBINED PREDICTION
saved_combined_model_path = os.path.join(
    os.getcwd(), 'models', 'best_combined_model.pt')
combined_model = CombinedModel(
    saved_combined_model_path, saved_img_model_path, saved_txt_model_path,
    decoder, transformer_img, EMBED_DIM, word_kernel_size=3)
combined_model.eval()

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

    

