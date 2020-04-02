from multiprocessing import Process, Queue


def reader(file, queue):
    while True:
        with open(file, 'rt', encoding='utf-8') as fin:
            while True:
                line = fin.readline()
                if not line:
                    queue.put(None)
                    break
                words = line[:-1].split()
                queue.put(words)


class SampleGenerator():
    def __init__(self, file, buffer_size):
        self.queue = Queue(buffer_size)
        self.reader = Process(target=reader, args=(file, self.queue))

    def start(self):
        self.reader.start()

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.queue.get()
        if sample is None:
            raise StopIteration()
        return sample

    def stop(self):
        self.reader.terminate()


if __name__ == '__main__':
    train_set = SampleGenerator('data/train_set', 100)
    train_set.start()
    for data in train_set:
        print(data)
    for data in train_set:
        print(data)
    train_set.stop()
