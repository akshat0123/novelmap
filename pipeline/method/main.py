from preprocessor import PreProcessor
from topic_model import TopicModel
from filepaths import *
import pickle



def main():

    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, 0)
    dictionary, corpus, books = preprocessor.get_library_info()

    books = [(book, books[book]['book_tokens']) for book in books]
    titles, books = zip(*books)

    tm = TopicModel()
    tm.add_documents(books)
    y_axes = tm.gen_y_axes(1000, 10)
    topic_freqs = tm.calc_topics(books, y_axes)

    sorted_topics = [(topic, topic_freqs[topic]) for topic in topic_freqs]
    sorted_topics = sorted(sorted_topics, key=lambda x: x[1], reverse=True)[:40]

    for pair in sorted_topics: print(pair)


if __name__ == '__main__':
    main()
