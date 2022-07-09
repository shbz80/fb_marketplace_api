import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

class ImageModel(torch.nn.Module):
    """This implements the image model class for using with the actual API."""
    def __init__(self, saved_model_path, decoder, transformer) -> None:
        super().__init__()
        # define and load the image model
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
        # use transformation for the image input
        input = self.transformer.transform(input)
        return self.img_model(input.unsqueeze(0))

    def predict(self, input):
        """Returns the category (str) of the image prediction """
        i = np.argmax(self.predict_proba(input)).item()
        return self.decoder[i]

    def predict_proba(self, input):
        """Returns the probabilities of the image prediction """
        output = self.forward(input).squeeze()
        return F.softmax(output, dim=0).detach()

    def predict_classes(self, input):
        """Returns the ranked category (str) list of the image prediction """
        proba = self.predict_proba(input)
        proba_args = proba.argsort().numpy()[::-1]
        classes = [self.decoder[arg] for arg in list(proba_args)]
        return classes
