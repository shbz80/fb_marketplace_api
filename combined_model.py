from email.policy import strict
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from text_model import TextClassification


class CombinedImageTextModel(nn.Module):
    def __init__(self, image_model, text_model, num_classes):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.num_classes = num_classes
        self.combined_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=self.num_classes),
            )

    def forward(self, ip_img, ip_txt):
        op_img = self.image_model(ip_img)
        op_txt = self.text_model(ip_txt).unsqueeze(0)
        op = torch.concat((op_img, op_txt), dim=1)
        return self.combined_layer(op)

class CombinedModel(torch.nn.Module):
    def __init__(
        self, saved_combined_model_path, saved_img_model_path, 
        saved_txt_model_path, decoder, img_transformer, word_embd_dim,
        word_kernel_size=2
        ) -> None:
        super().__init__()
        # define and load the image model
        self.img_model = models.resnet50(pretrained=True)
        num_ftrs = self.img_model.fc.in_features
        self.img_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                          nn.ReLU(),
                                          )
        self.img_model.load_state_dict(
            torch.load(saved_img_model_path,
                       map_location=torch.device('cpu')), strict=False
        )
        self.decoder = decoder
        self.img_transformer = img_transformer

        # define and load the text model
        self.word_embd_dim = word_embd_dim
        self.word_kernel_size = word_kernel_size
        self.txt_model = TextClassification(
            word_embd_dim, len(decoder), word_kernel_size=word_kernel_size)
        # get the current classification layers
        linear_layers = list(self.txt_model.children())[-1]
        # define the new classification layers by droping the last liner layer
        new_linear_layers = nn.Sequential(*list(linear_layers.children())[:-1])
        # redefine the model by replacing the classification layers
        self.txt_model.linear_layer = new_linear_layers
        self.txt_model.load_state_dict(
            torch.load(saved_txt_model_path,
                       map_location=torch.device('cpu')), strict=False
        )
        # load the combined model
        self.combined_model = CombinedImageTextModel(
            self.img_model, self.txt_model, len(decoder))
        self.combined_model.load_state_dict(
            torch.load(saved_combined_model_path,
                       map_location=torch.device('cpu'))
        )
        # freeze all params
        for param in self.combined_model.parameters():
            param.requires_grad = False
        
    def forward(self, img_ip, txt_ip):
        img_ip = self.img_transformer.transform(img_ip)
        return self.combined_model(img_ip.unsqueeze(0), txt_ip.unsqueeze(0))

    def predict(self, img_ip, txt_ip):
        i = np.argmax(self.predict_proba(img_ip, txt_ip)).item()
        return self.decoder[i]

    def predict_proba(self, img_ip, txt_ip):
        output = self.forward(img_ip, txt_ip).squeeze()
        return F.softmax(output, dim=0).detach()

    def predict_classes(self, img_ip, txt_ip):
        proba = self.predict_proba(img_ip, txt_ip)
        proba_args = proba.argsort().numpy()[::-1]
        classes = [self.decoder[arg] for arg in list(proba_args)]
        return classes
