from goodreads_shelves import *
from preprocessing import *

BOOK_TITLES_PATH = '../data/base/books.txt'
BOOK_SHELVES_PATH = '../data/base/book_shelves.json'
DICTIONARY_PATH = '../data/dumps/book_dictionary.dict'


def main():

    # Get Goodreads shelves for books for evaluation
    get_shelves_given_titles(BOOK_TITLES_PATH, BOOK_SHELVES_PATH)

    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    stopwords = set(stopwords.words('english'))

    preprocessor = PreProcessing(tokenizer, lemmatizer, stopwords, BOOK_DIRECTORY)
    corpus, dictionary = preprocessor.process_books()


if __name__ == '__main__': main()
