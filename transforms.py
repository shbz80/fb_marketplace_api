from PIL import Image

class TransformImage():
    def __init__(self, saved_transformer, img_preprocessor) -> None:
        self.saved_transformer = saved_transformer
        self.size = (224, 224)
        self.mode = 'RGB'
        self.img_preprocessor = img_preprocessor

    def transform(self, im):
        preprocessed_im = self.img_preprocessor(im, size=(224, 224),
                                                mode='RGB')
        transformed_im = self.saved_transformer(preprocessed_im)
        return transformed_im
