from model.word2vec_model import Word2VecModel

if __name__ == '__main__':
    model = Word2VecModel.load_model("data/test_model")
    similar_words = model.find_n_similar("市场", 5)
    for score, word in similar_words:
        print("{:.2f} {:s}".format(score, word))
    model.save_vector("data/vector.txt")