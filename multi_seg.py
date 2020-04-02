import jieba
import time
import os
from multiprocessing import Process

stop_words_file = 'data/cn_stopwords.txt'
output_file = 'data/multi_seg_sents'
input_file = 'data/sentences'
n_process = 4


def cut_sentence(file_name, start_pos, end_pos, stop_words, save_file):
    with open(file_name, "rt", encoding="utf-8") as fin:
        with open(save_file, "wt", encoding="utf-8") as fout:
            for i in range(end_pos):
                line = fin.readline()[:-1]
                if i < start_pos:
                    continue
                words = jieba.cut(line)
                filter_words = [
                    word for word in words
                    if word != " " and word not in stop_words
                ]
                if len(filter_words) > 2:
                    print(" ".join(filter_words), file=fout)


def seg_file(file_name, n):
    pos_list = []
    file_size = 0
    with open(file_name, "rt", encoding="utf-8") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            file_size += 1
    block_size = file_size // n
    if block_size % n != 0:
        block_size += 1
    start_pos = 0
    for i in range(n):
        end_pos = min(start_pos + block_size, file_size)
        pos_list.append((start_pos, end_pos))
        start_pos += block_size
    return pos_list


def combine_file(output_file, n):
    with open(output_file, 'wt', encoding="utf-8") as fout:
        for i in range(n):
            with open(output_file + str(i), 'rt', encoding='utf-8') as fin:
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    print(line[:-1], file=fout)
            os.remove(output_file + str(i))


def read_stop_words(file):
    stop_words = set()
    with open(file, 'rt', encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line[:-1]
            stop_words.add(line)
    return stop_words


if __name__ == '__main__':
    start_time = time.time()
    stop_words = read_stop_words(stop_words_file)
    pos_list = seg_file(input_file, n_process)
    processes = []
    for i in range(n_process):
        start_pos, end_pos = pos_list[i]
        process = Process(target=cut_sentence,
                          args=(input_file, start_pos, end_pos, stop_words,
                                output_file + str(i)))
        processes.append(process)
        process.start()
    for i in range(n_process):
        processes[i].join()
    combine_file(output_file, n_process)
    end_time = time.time()
    print("cost time:", end_time - start_time)
