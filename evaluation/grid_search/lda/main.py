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


    # lda 
    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, min_count = 10, headword_removal_perc = 0.05)
    dictionary, corpus, books = preprocessor.get_library_info()

    num_topics = 2000
    chunksize = 10
    passes = 5

    start_time = time.time()

    lda = Library(dictionary, corpus, num_topics = num_topics, chunksize = chunksize, model_type = 'LDA', passes = passes)
    for title in tqdm(books):
        lda.add_book(books[title])

    pca = PCA(n_components=2) 

    book_vectors = []
    titles = []
    for title in books:
        vector = np.zeros(50)
        book = lda.model[dictionary.doc2bow(books[title]['book_tokens'])]
        for pair in book:
            vector[pair[0]] = pair[1]
        titles.append(title)
        book_vectors.append(vector)

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


if __name__ == '__main__':
    main()
