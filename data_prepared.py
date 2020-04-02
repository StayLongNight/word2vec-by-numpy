import re
from dictionary import Dictionary

xml_file = "data/news_sohusite.xml"
sentence_file = "data/origin_sentence"
seg_sent_file = "data/seg_sents"
filter_seg_sent_file = "data/filter_seg_sents"
train_file = "data/train_set"
stop_words_file = "data/cn_stopwords.txt"
dict_file = "data/dict"
windows_size = 3


def cut_paragraph(content):
    sentences = []
    sent_delimiter = r'(。|！|\!|\?|？)'
    retain_reg = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    new_para = re.sub(sent_delimiter, '\n', content)
    for sentence in new_para.split('\n'):
        filter_sent = re.sub(retain_reg, " ", sentence)
        if re.search(r"[^ ]", filter_sent):
            sentences.append(filter_sent)
    return sentences


def parse_xml(line):
    sentences = []
    if line.find("<contenttitle>") != -1:
        line = line.replace("<contenttitle>", "")
        line = line.replace("</contenttitle>", "")
        sentences = cut_paragraph(line)
    elif line.find("<content>") != -1:
        line = line.replace("<content>", "")
        line = line.replace("</content>", "")
        sentences = cut_paragraph(line)
    return sentences


def sent_to_train_set(sentence, c=windows_size):
    samples = []
    words = sentence.split()
    for i, word in enumerate(words):
        sample = [word] + words[max(i - c, 0):i] + words[i + 1:i + 1 + c]
        samples.append(" ".join(sample))
    return samples


def get_filter(dictionary):
    def word_filter(sentence):
        sentences = []
        words = sentence.split()
        filter_words = [word for word in words if dictionary.exist(word)]
        if len(filter_words) > 2:
            sentences.append(" ".join(filter_words))
        return sentences

    return word_filter


def data_transform(input_file, output_file, processor, n=-1):
    cnt = 0
    with open(input_file, 'rt', encoding='utf-8') as fin:
        with open(output_file, 'wt', encoding='utf-8') as fout:
            while True:
                sentence = fin.readline()
                if not sentence:
                    break
                sentence = sentence[:-1]
                sentences = processor(sentence)
                if sentences != []:
                    for sent in sentences:
                        print(sent, file=fout)
                if n > 0:
                    cnt += 1
                    if cnt == n:
                        break


def read_sentences(file):
    sentences = []
    with open(file, 'rt', encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line[:-1]
            sentences.append(line)
    return sentences


def get_word_freq(input_file, output_file):
    sentences = []
    word_cnt = {}
    with open(input_file, 'rt', encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            sentences.append(line[:-1])
    for sentence in sentences:
        for word in sentence.split():
            if word not in word_cnt:
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1
    with open(output_file, 'wt', encoding='utf-8') as fout:
        for word, cnt in word_cnt.items():
            print(word, ' ', cnt, file=fout)


if __name__ == '__main__':
    #data_transform(xml_file, sentence_file, parse_xml)
    '''
    corpus = read_sentences(seg_sent_file)
    dictionary = Dictionary(corpus=corpus,
                            max_freq=0.5,
                            min_freq=3,
                            voc_size=10000)
    dictionary.save_dict(dict_file)
    '''
    dictionary = Dictionary(file=dict_file)
    word_filter = get_filter(dictionary)
    data_transform(seg_sent_file, filter_seg_sent_file, word_filter, 500000)
    data_transform(filter_seg_sent_file, train_file, sent_to_train_set)
    get_word_freq(filter_seg_sent_file, 'data/word_cnt')
