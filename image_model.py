import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

class ImageModel(torch.nn.Module):
    def __init__(self, saved_model_path, decoder, transformer) -> None:
        super().__init__()
        self.img_model = models.resnet50(pretrained=True)
        num_ftrs = self.img_model.fc.in_features
        self.img_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                          nn.ReLU(),
                                          nn.Linear(256, len(decoder)),
                                          )
        self.img_model.load_state_dict(torch.load(
            saved_model_path, map_location=torch.device('cpu')))
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
