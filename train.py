from dictionary import Dictionary
from model.word2vec_train import Word2Vec, save_model
from model.word2vec_model import Word2VecModel
from sample import SampleGenerator
import sys

sys.setrecursionlimit(1000000)
lr = 0.01
buffer_size = 1000
model_path = 'data/cbow_neg_model'

if __name__ == '__main__':
    train_set = SampleGenerator('data/train_set', buffer_size)
    train_set.start()
    dictionary = Dictionary(file='data/dict')
    trainer = Word2Vec(dictionary,
                       dim=100,
                       model_type='CBOW',
                       optimize_type='NEG_SAMPLE')
    trainer.train(train_set, 5, lr, model_path)
    vector_model = trainer.get_vector_model()
    vector_model.save_model(model_path)
    train_set.stop()
