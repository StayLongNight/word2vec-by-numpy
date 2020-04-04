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
