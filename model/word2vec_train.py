import numpy as np
import pickle
import time
from model.optimize_model import NegSample, HierachicalSoftmax
from model.word2vec_model import Word2VecModel


def save_model(obj, model_file):
    with open(model_file, 'wb') as fout:
        pickle.dump(obj, fout)


def load_model(model_file):
    model = None
    with open(model_file, 'rb') as fin:
        model = pickle.load(fin)
    return model


class Word2Vec():
    def __init__(self, dictionary, dim, model_type, optimize_type):
        self.dict = dictionary
        self.dim = dim
        self.voc_size = self.dict.size()
        self.W = np.random.rand(self.voc_size, dim)
        self.model_type = model_type
        self.optimize_type = optimize_type
        if optimize_type == 'NEG_SAMPLE':
            self.optimizer = NegSample(word_cnt_file='data/word_cnt',
                                       dim=self.dim,
                                       dictionary=self.dict)
        elif optimize_type == 'HIERACHICAL_SOFTMAX':
            print(1)
            self.optimizer = HierachicalSoftmax(word_cnt_file='data/word_cnt',
                                                dim=self.dim,
                                                dictionary=self.dict)

    def backward(self, center, windows, lr):
        if self.model_type == 'SKIP_GRAM':
            center_pos = self.dict(center)
            ys = [self.dict(word) for word in windows]
            x = self.W[center_pos]
            for y in ys:
                delta_x = self.optimizer.backward(x, y, lr)
                x -= lr * delta_x
        elif self.model_type == 'CBOW':
            y_id = self.dict(center)
            windows_pos = [self.dict(word) for word in windows]
            windows_vecs = self.W[windows_pos]
            windows_size = windows_vecs.shape[0]
            x = windows_vecs.sum(axis=0) / windows_size
            delta_x = self.optimizer.backward(x, y_id, lr)
            windows_vecs -= delta_x * lr / windows_size

    def train(self, train_data, epoch, lr, model_file=None):
        last_time = time.time()
        for i in range(epoch):
            for j, data in enumerate(train_data):
                center = data[0]
                windows = data[1:]
                self.backward(center, windows, lr)
                if j % 1000 == 0:
                    cur_time = time.time()
                    print("epoch {} {} cost time {:.2f}s".format(
                        i, j, cur_time - last_time))
                    cur_time = last_time

    def get_vector_model(self):
        model = Word2VecModel(self.W, self.dict)
        return model