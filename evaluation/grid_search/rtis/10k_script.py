from goodreads_shelves import get_shelves_given_titles
from sklearn.neighbors import KNeighborsClassifier
from preprocessor import PreProcessor
from sklearn.decomposition import PCA
from topic_model import TopicModel
from library import Library
from os.path import isfile
from tqdm import tqdm
import pickle, time
import numpy as np
import json

TITLES = './data/base/books.txt'
SHELVES = './data/base/book_shelves.json'

DELIM = './data/base/books_delimiter_data.txt'
RAW = './data/raw'

DICT = './data/dumps/book_dictionary.dict'
CORP = './data/dumps/book_corpus.dict'
TOKEN = './data/dumps/token_dump.p'


def main():

    # Baseline

    baseline_book_vectors = json.load(open(SHELVES, 'r'))
    baseline_book_vectors = {title: baseline_book_vectors[title] for title in baseline_book_vectors if len(baseline_book_vectors[title]) != 0}
    del(baseline_book_vectors['The Merry Adventures of Robin Hood'])

    vocab = []
    for title in baseline_book_vectors:
        vocab += baseline_book_vectors[title]

    vocab = enumerate(list(set(vocab)))
    vocab_dict = { pair[1]: pair[0] for pair in vocab }

    pca = PCA(n_components=2)

    book_matrix = []
    titles = []
    for title in baseline_book_vectors:
        book_vector = np.zeros(len(vocab_dict)) 
        for word in baseline_book_vectors[title]:
            book_vector[vocab_dict[word]] += 1
        titles.append(title)
        book_matrix.append(book_vector)
        
    book_matrix = np.array(book_matrix)
    pca_books = np.array([book for book in pca.fit_transform(book_matrix)])

    x, y = zip(*pca_books)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(pca_books, np.zeros(len(x)))
    neighbors = [book for book in knn.kneighbors(pca_books, n_neighbors=5)[1]]

    base_neighbor_names = {} 
    for group in neighbors:
        base_neighbor_names[titles[group[0]]] = [titles[j] for j in group[1:]]


    # rtis

    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, min_count = 10, headword_removal_perc = 0.05)
    dictionary, corpus, books = preprocessor.get_library_info()

    books = [(book, books[book]['book_tokens']) for book in books]
    titles, books = zip(*books)

    y_axis_count = 10000

    for y_axis_length in [10, 25, 40]: 

        for threshold in [0.7, 0.8, 0.9, 1.0]:

            for unigram in [True, False]:

                for iteration in [1, 2, 3]:

                    start_time = time.time()
                    tm = TopicModel(y_axis_count, y_axis_length, threshold)
                    tm.add_documents(books)
                    y_axes = tm.gen_y_axes()

                    if unigram == True: 
                        topic_freqs, book_topics = tm.calc_unigram_topics(books, y_axes)

                    else:
                        topic_freqs, book_topics = tm.calc_trigram_topics(books, y_axes)

                    for num_topics in range(1000, 6000, 1000):

                        sorted_topics = [(topic, topic_freqs[topic]) for topic in topic_freqs]
                        sorted_topics = sorted(sorted_topics, key=lambda x: x[1], reverse=True)
                        top_percent_count = int(len(sorted_topics) * .05)
                        sorted_topics = sorted_topics[top_percent_count:top_percent_count + num_topics] 

                        book_vectors = []
                        for book_index in range(len(book_topics)):
                            book_vector = np.zeros(len(sorted_topics))

                            for topic_index in range(len(sorted_topics)):
                                topic = sorted_topics[topic_index][0]

                                if topic in book_topics[book_index]:
                                    book_vector[topic_index] = np.average(book_topics[book_index][topic])

                            book_vector_total = np.sum(book_vector)
                            book_vector = [point/book_vector_total for point in book_vector]
                            book_vectors.append(book_vector)
                            
                        book_matrix = np.array(book_vectors)
                        pca_books = np.array([book for book in pca.fit_transform(book_matrix)])

                        x, y = zip(*pca_books)

                        knn = KNeighborsClassifier(n_neighbors=5)
                        knn.fit(pca_books, np.zeros(len(x)))
                        neighbors = [book for book in knn.kneighbors(pca_books, n_neighbors=5)[1]]

                        neighbor_names = {} 
                        for group in neighbors:
                            neighbor_names[titles[group[0]]] = [titles[j] for j in group[1:]]
                              
                        total_count = 0
                        for title in neighbor_names:
                            count = 0
                            for base_title in base_neighbor_names[title]:
                                if base_title in neighbor_names[title]:
                                    count += 1
                            total_count += count
                            
                        accuracy = total_count/(len(neighbor_names)*4)
                        total_time = time.time() - start_time

                        print('%s,%s,%s,%s,%s,%s,%s,%s' % (y_axis_count, y_axis_length, threshold, unigram, num_topics, iteration, accuracy, total_time))



if __name__ == '__main__':
    main()
