from goodreads_shelves import get_shelves_given_titles
from preprocessor import Preprocessor
from library import Library
from tqdm import tqdm


TITLES = '../data/base/books.txt'
SHELVES = '../data/base/book_shelves.json'

DELIM = '../data/base/books_delimiter_data.txt'
RAW = '../data/raw'

DICT = '../data/dumps/book_dictionary.dict'
CORP = '../data/dumps/book_corpus.dict'
TOKEN = '../data/dumps/token_dump.p'


def main():

    # Get Goodreads shelves for books for evaluation
    # get_shelves_given_titles(TITLES, SHELVES)

    # Get book texts split into chapters
    preprocessor = Preprocessor(DICT, CORP, DELIM, RAW, TOKEN, 0)
    dictionary, corpus, books = preprocessor.get_library_info()

    # Add all books into library 
    tfidf = Library(dictionary, corpus, 50, 'TFIDF')
    for title in tqdm(books): tfidf.add_book(books[title])

    lda = Library(dictionary, corpus, 50, 'LDA')
    for title in tqdm(books): lda.add_book(books[title])

    lsi = Library(dictionary, corpus, 50, 'LSI')
    for title in tqdm(books): lsi.add_book(books[title])

    # topics_tfidf = tfidf.get_topics('text')
    topics_lda = lda.get_topics('text')
    topics_lsi = lsi.get_topics('text')

    # print(topics_tfidf) 
    print(topics_lda) 
    print(topics_lsi) 

if __name__ == '__main__':
    main()
