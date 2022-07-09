from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import torch
import numpy as np
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

class TransformText():
    def __init__(self, word_to_idx, embeddings, max_word_len, embd_dim) -> None:
        self.word_to_idx = word_to_idx
        self.embeddings = embeddings
        self.max_word_len = max_word_len
        self.embd_dim = embd_dim

    def preprocess(self, txt):
        if not isinstance(txt, str):
            raise ValueError('Input not a string.')
        # convert all tolower case
        lower_txt = txt.lower()
        # remove everthing except alphanumeric chars
        word_tokenize = RegexpTokenizer(r'\w+')
        tokenized_lower_txt = word_tokenize.tokenize(lower_txt)
        # remove stopwords
        stop_words = stopwords.words('english')
        stop_removed_tokenized_lower_txt = [
            word for word in tokenized_lower_txt if word not in stop_words]
        # lemmtize words
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_stop_removed_tokenized_lower_txt = [wordnet_lemmatizer.lemmatize(
            word) for word in stop_removed_tokenized_lower_txt]
        # stem words
        snowball_stemmer = SnowballStemmer('english')
        stemmed_lemmatized_stop_removed_tokenized_lower_txt = [snowball_stemmer.stem(
            word) for word in lemmatized_stop_removed_tokenized_lower_txt]
        # the final processed text
        processed_txt = stemmed_lemmatized_stop_removed_tokenized_lower_txt
        return processed_txt

    def get_word_to_indx(self, words):
        if not isinstance(words, list):
            raise TypeError('Input not a list.')
        idxs = []
        for word in words:
            word_idx = self.word_to_idx.get(word)
            if not word_idx is None:
                idxs.append(word_idx)
        if not idxs:
            raise ValueError('Empty description!')
        return idxs

    def get_idxs_to_embedding_seq(self, idxs):
        idxs = torch.from_numpy(np.array(idxs).reshape(-1, 1))
        embds = self.embeddings(idxs).squeeze()
        embds_seq = torch.concat([emb for emb in embds])
        return embds_seq

    def pad_embedding_seq(self, embds_seq):
        pad_len = self.max_word_len * self.embd_dim - len(embds_seq)
        assert(pad_len >= 0)
        pad_len_r = pad_len // 2
        pad_len_l = pad_len - pad_len_r
        pad_r = torch.zeros(pad_len_r)
        pad_l = torch.zeros(pad_len_l)
        padded_embds_seq = torch.concat([pad_l, embds_seq, pad_r])
        return padded_embds_seq

    def transform(self, txt):
        input = self.preprocess(txt)
        input = self.get_word_to_indx(input)
        input = self.get_idxs_to_embedding_seq(input)
        input = self.pad_embedding_seq(input)
        return input


    


        

        
