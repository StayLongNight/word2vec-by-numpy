from queue import PriorityQueue
import numpy as np
import pickle


class Word2VecModel():
    def __init__(self, W, dictionary):
        self.W = W
        self.dict = dictionary

    def get_word_vec(self, word):
        ids = self.dict(word)
        vec = self.W[ids]
        return vec

    def find_n_similar(self, word, n):
        ids = self.dict(word)
        x = self.W[ids]
        que = PriorityQueue()
        for i, y in enumerate(self.W):
            if i != ids:
                score = np.dot(y, x)
                if que.qsize() < n:
                    que.put((score, i))
                else:
                    top = que.get()
                    if top[0] < score:
                        que.put((score, i))
                    else:
                        que.put(top)
        word_list = []
        while que.qsize() > 0:
            score, ids = que.get()
            word_list.append((score, self.dict.get_word(ids)))
        return word_list

    def save_model(self, model_file):
        with open(model_file, 'wb') as fout:
            pickle.dump(self, fout)

    def save_vector(self, file):
        with open(file, 'wt', encoding='utf-8') as fout:
            voc_size, dim = self.W.shape
            for i in range(voc_size):
                print(self.dict.get_word(i), end='\t', file=fout)
                for j in range(dim):
                    print(self.W[i, j], end='\t', file=fout)
                print(file=fout)

    @staticmethod
    def load_model(model_file):
        model = None
        with open(model_file, 'rb') as fin:
            model = pickle.load(fin)
        return model