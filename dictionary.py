class Dictionary():
    def __init__(self,
                 file=None,
                 corpus='',
                 max_freq=1,
                 min_freq=0,
                 voc_size=-1):
        self.word_to_id = {}
        self.id_to_word = {}
        if file is not None:
            with open(file, 'rt', encoding='utf-8') as fin:
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    line = line[:-1]
                    word, ids = line.split()
                    ids = int(ids)
                    self.word_to_id[word] = ids
                    self.id_to_word[ids] = word
        else:
            total_words = 0
            word_cnt = {}
            filter_word_cnt = []
            for sentence in corpus:
                words = sentence.split()
                for word in words:
                    if word not in word_cnt:
                        word_cnt[word] = 1
                    else:
                        word_cnt[word] += 1
                    total_words += 1
            for word, cnt in word_cnt.items():
                if cnt >= min_freq and cnt / total_words < max_freq:
                    filter_word_cnt.append((word, cnt))
            if voc_size > 0:
                filter_word_cnt.sort(key=lambda x: x[1], reverse=True)
                filter_word_cnt = filter_word_cnt[0:voc_size]
            ids = 0
            for word, cnt in filter_word_cnt:
                self.word_to_id[word] = ids
                self.id_to_word[ids] = word
                ids += 1

    def save_dict(self, file_path):
        with open(file_path, 'wt', encoding="utf-8") as fout:
            for word, ids in self.word_to_id.items():
                print(word, ' ', ids, file=fout)

    def __call__(self, word):
        return self.word_to_id[word]

    def get_word(self, ids):
        return self.id_to_word[ids]

    def exist(self, word):
        return word in self.word_to_id

    def size(self):
        return len(self.word_to_id)
