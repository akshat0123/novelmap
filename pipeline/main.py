from goodreads_shelves import get_shelves_given_titles
from preprocessor import Preprocessor


BOOK_TITLES_PATH = '../data/base/books.txt'
BOOK_SHELVES_PATH = '../data/base/book_shelves.json'

DELIM = '../data/base/books_delimiter_data.txt'
RAW = '../data/raw'

DICT = '../data/dumps/book_dictionary.dict'
CORP = '../data/dumps/book_corpus.dict'
TOKEN = '../data/dumps/token_dump.p'


def main():

    # Get Goodreads shelves for books for evaluation
    # get_shelves_given_titles(BOOK_TITLES_PATH, BOOK_SHELVES_PATH)

    # Get book texts split into chapters
    preprocessor = Preprocessor(DICT, CORP, DELIM, RAW, TOKEN, 0)
    dictionary, corpus, books = preprocessor.get_library_info()


if __name__ == '__main__':
    main()
