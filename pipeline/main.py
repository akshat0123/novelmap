from goodreads_shelves import get_shelves_given_titles
from preprocessor import PreProcessor
from library import Library
from tqdm import tqdm
from os.path import isfile
import pickle

TITLES = '../data/base/books.txt'
SHELVES = '../data/base/book_shelves.json'

DELIM = '../data/base/books_delimiter_data.txt'
RAW = '../data/raw'

DICT = '../data/dumps/book_dictionary.dict'
CORP = '../data/dumps/book_corpus.dict'
TOKEN = '../data/dumps/token_dump.p'

LDA_MODEL = '../data/dumps/lda_model.p'
LSI_MODEL = '../data/dumps/lsi_model.p'


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


    if not isfile(LSI_MODEL):
        lsi = Library(dictionary, corpus, num_topics = 50, chunksize=1000, model_type = 'LSI')
        for title in tqdm(books):
            lsi.add_book(books[title])
        
        pickle.dump(lsi, open(LSI_MODEL, 'wb'))
    else:
        lsi = pickle.load(open(LSI_MODEL, 'rb'))


if __name__ == '__main__':
    main()
