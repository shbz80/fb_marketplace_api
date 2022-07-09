import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class TextClassification(nn.Module):
    """This is the original class for the text
    classification. This will be used to load the trained (saved) model.
    TODO: Code duplication! Import it from the original source.
    """
    def __init__(self, word_embd_dim, num_classes, word_kernel_size=2):
        super().__init__()
        self.word_embd_dim = word_embd_dim
        self.word_kernel_size = word_kernel_size
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=48,
                      kernel_size=self.word_kernel_size * self.word_embd_dim,
                      stride=self.word_embd_dim,
                      padding=(self.word_kernel_size - 1) * self.word_embd_dim),
            nn.Tanh(),
            nn.Flatten(start_dim=0),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=63408, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, inputs):
        x = self.cnn_layers(inputs)
        return self.linear_layer(x)

class TextModel(nn.Module):
    def __init__(self, saved_txt_model_path, word_embd_dim, decoder, word_kernel_size=2):
        super().__init__()
        self.word_embd_dim = word_embd_dim
        self.word_kernel_size = word_kernel_size
        self.decoder = decoder
        self.text_model = TextClassification(
            word_embd_dim, len(decoder), word_kernel_size=word_kernel_size)
        self.text_model.load_state_dict(
            torch.load(saved_txt_model_path, map_location=torch.device('cpu'))
            )
        # freeze all params
        for param in self.text_model.parameters():
            param.requires_grad=False

    def forward(self, inputs):
        x = self.text_model(inputs)
        return x

    def predict(self, input):
        i = np.argmax(self.predict_proba(input)).item()
        return self.decoder[i]

    def predict_proba(self, input):
        output = self.forward(input.unsqueeze(0)).squeeze()
        return F.softmax(output, dim=0).detach()

    def predict_classes(self, input):
        proba = self.predict_proba(input)
        proba_args = proba.argsort().numpy()[::-1]
        classes = [self.decoder[arg] for arg in list(proba_args)]
        return classes



