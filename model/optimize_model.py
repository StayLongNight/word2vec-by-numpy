import numpy as np
import heapq
import random


def read_word_cnt(file):
    word_cnt = {}
    with open(file, 'rt', encoding='utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line[:-1]
            word, cnt = line.split()
            word_cnt[word] = int(cnt)
    return word_cnt


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def size(self):
        return len(self._queue)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid():
    def __init__(self, dim):
        self.w = np.random.rand(dim)
        self.b = np.random.rand(1)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        z = np.dot(self.w, x) + self.b
        return sigmoid(z)

    def backward(self, x, y, lr):
        delta_z = self.forward(x) - y
        delta_w = delta_z * x
        delta_x = delta_z * self.w
        self.w = self.w - lr * delta_w
        self.b = self.b - lr * delta_z
        return delta_x


class HuffNode():
    def __init__(self,
                 ids=None,
                 weight=None,
                 parent=None,
                 left=None,
                 right=None,
                 dim=None):
        self.id = ids
        self.weight = weight
        self.parent = parent
        self.left = left
        self.right = right
        self.sigmoid = None
        if self.id is None:
            self.sigmoid = Sigmoid(dim)


class HierachicalSoftmax():
    def __init__(self,
                 word_cnt=None,
                 word_cnt_file=None,
                 dim=None,
                 dictionary=None):
        self.root = None
        self.dict = dictionary
        self.id_pointer = [None for i in range(self.dict.size())]
        if word_cnt_file is not None:
            word_cnt = read_word_cnt(word_cnt_file)
        if word_cnt is not None:
            self.build_tree(word_cnt, dim)

    def forward(self, x):
        cur = self.root
        while cur.id is None:
            z = cur.sigmoid(x)
            if z > 0.5:
                cur = cur.left
            else:
                cur = cur.right
        return self.dict[cur.id]

    def backward(self, x, y_id, lr):
        delta_x = 0
        child = self.id_pointer[y_id]
        while child.parent is not None:
            parent = child.parent
            d = 0
            if parent.left == child:
                d = 1
            delta_x += parent.sigmoid.backward(x, d, lr)
            child = parent
        return delta_x

    def build_tree(self, word_cnt, dim):
        que = PriorityQueue()
        for word, cnt in word_cnt.items():
            ids = self.dict(word)
            node = HuffNode(ids, cnt)
            self.id_pointer[ids] = node
            que.push(node, cnt)
        while que.size() > 1:
            top1 = que.pop()
            top2 = que.pop()
            if top1.weight < top2.weight:
                temp = top1
                top1 = top2
                top2 = temp
            new_weight = top1.weight + top2.weight
            new_node = HuffNode(None, new_weight, None, top1, top2, dim)
            top1.parent = new_node
            top2.parent = new_node
            que.push(new_node, new_weight)
        self.root = que.pop()


class NegSample():
    def __init__(self,
                 word_cnt=None,
                 word_cnt_file=None,
                 dim=None,
                 dictionary=None,
                 neg_num=5):
        self.sample_array = []
        self.dim = dim
        self.dict = dictionary
        self.voc_size = self.dict.size()
        self.sigmoid_array = [Sigmoid(dim) for i in range(self.voc_size)]
        self.total_freq = 0
        self.neg_num = neg_num
        if word_cnt_file is not None:
            word_cnt = read_word_cnt(word_cnt_file)
        if word_cnt is not None:
            self.make_sample_array(word_cnt)

    def make_sample_array(self, word_cnt):
        for word, cnt in word_cnt.items():
            sample_freq = int(cnt**(0.75)) + 1
            self.total_freq += sample_freq
            for i in range(sample_freq):
                self.sample_array.append(self.dict(word))

    def forward(self, x):
        values = [sigmoid_i.forward(x) for sigmoid_i in self.sigmoid_array]
        return self.dict.get_word(values.index(max(values)))

    def backward(self, x, y_id, lr):
        neg_samples = self.get_neg_samples()
        delta_x = self.sigmoid_array[y_id].backward(x, 1, lr)
        for pos in neg_samples:
            delta_x += self.sigmoid_array[pos].backward(x, 0, lr)
        return delta_x

    def get_neg_samples(self):
        sample_pos = [
            random.randint(0, self.total_freq - 1) for i in range(self.neg_num)
        ]
        neg_samples = [self.sample_array[i] for i in sample_pos]
        return neg_samples
