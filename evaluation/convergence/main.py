from preprocessor import PreProcessor
from topic_model import TopicModel
from tqdm import trange
from filepaths import *
from tqdm import tqdm
import numpy as np
import pickle



def main():

    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, min_count = 10, headword_removal_perc = 0.05)
    dictionary, corpus, books = preprocessor.get_library_info()

    books = [(book, books[book]['book_tokens']) for book in books]
    titles, books = zip(*books)


    for y_axis_count in tqdm(range(10000, 50000, 10000)):
    # for y_axis_count in tqdm(range(50000, 80000, 10000)):
    # for y_axis_count in tqdm(range(80000, 100000, 10000)):
    # for y_axis_count in tqdm(range(100000, 110000, 10000)):

        for i in trange(10):

            tm = TopicModel(y_axis_count, 25, 1.0)
            tm.add_documents(books)

            y_axes = tm.gen_y_axes()
            topic_freqs, book_topics = tm.calc_unigram_topics(books, y_axes)

            sorted_topics = [(topic, topic_freqs[topic]) for topic in topic_freqs]
            sorted_topics = sorted(sorted_topics, key=lambda x: x[1], reverse=True)
            top_percent_count = int(len(sorted_topics) * .05)
            sorted_topics = sorted_topics[top_percent_count:top_percent_count + 2000]

            for topic in sorted_topics:
                print('%s|%s|%s' % (y_axis_count, i, topic))


if __name__ == '__main__':
    main()
