from preprocessor import PreProcessor
from topic_model import TopicModel
from filepaths import *
import numpy as np
import pickle



def main():

    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN)
    dictionary, corpus, books = preprocessor.get_library_info()

    books = [(book, books[book]['book_tokens']) for book in books]
    titles, books = zip(*books)

    tm = TopicModel(0.1)
    tm.add_documents(books)

    y_axes = tm.gen_y_axes(1000, 10)
    topic_freqs, book_topics = tm.calc_topics(books, y_axes)

    sorted_topics = [(topic, topic_freqs[topic]) for topic in topic_freqs]
    sorted_topics = sorted(sorted_topics, key=lambda x: x[1], reverse=True)[:20]
    for pair in sorted_topics: print(pair)

    # book_vectors = []
    # for book_index in range(len(book_topics)):
    #     book_vector = np.zeros(len(sorted_topics))

    #     for topic_index in range(len(sorted_topics)):
    #         topic = sorted_topics[topic_index][0]

    #         if topic in book_topics[book_index]:
    #             book_vector[topic_index] = np.min(book_topics[book_index][topic])

    #     book_vector_total = np.sum(book_vector)
    #     book_vector = [point/book_vector_total for point in book_vector]
    #     book_vectors.append(book_vector)

    # for pair in sorted_topics: print(pair)

    # for book_index in range(len(books)):
    #     title = titles[book_index]
    #     vector = book_vectors[book_index]
    #     print(title, vector)


if __name__ == '__main__':
    main()
