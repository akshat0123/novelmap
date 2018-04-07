from goodreads_shelves import get_shelves_given_titles
from preprocessor import PreProcessor
from library import Library
from tqdm import tqdm
from os.path import isfile
from gensim import similarities
import pickle

TITLES = '../data/base/books.txt'
SHELVES = '../data/base/book_shelves.json'

DELIM = '../data/base/books_delimiter_data.txt'
RAW = '../data/raw'

DICT = '../data/dumps/book_dictionary.dict'
CORP = '../data/dumps/book_corpus.dict'
TOKEN = '../data/dumps/token_dump.p'

LDA_MODEL = '../data/dumps/lda_model.p'


def main():

    # Get Goodreads shelves for books for evaluation
    # get_shelves_given_titles(TITLES, SHELVES)

    # Get book texts split into chapters
    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, 0)
    dictionary, corpus, books = preprocessor.get_library_info()
    
    if not isfile(LDA_MODEL):
        lda = Library(dictionary, corpus, num_topics = 50, chunksize=1000, model_type = 'LDA')
        for title in tqdm(books):
            lda.add_book(books[title])
        
        pickle.dump(lda, open(LDA_MODEL, 'wb'))
    else:
        lda = pickle.load(open(LDA_MODEL, 'rb'))

    topics = lda.get_topics('text')

    book_corpi = []
    for book in lda.books:
        corpus = book.book_corpus
        name = book.book_title
        book_corpi.append((name, corpus))
    
    lib_model = lda.model
    index = similarities.MatrixSimilarity(lib_model[lda.corpus])
    book_vec = lib_model[book_corpi[22][1]]
    print(book_corpi[22][0])
    print(book_corpi[34][0])
    print(sorted(list(enumerate(index[book_vec])), key=lambda item: -item[1]))



if __name__ == '__main__':
    main()
