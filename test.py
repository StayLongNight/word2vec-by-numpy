from model.word2vec_model import Word2VecModel

if __name__ == '__main__':
    model = Word2VecModel.load_model("data/cbow_neg_model")
    model.norm()
    similar_words = model.find_n_similar("日本", 5)
    for score, word in similar_words:
        print("{:s} 相似度:{:.6f} ".format(word, score))
