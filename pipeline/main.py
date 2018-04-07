from goodreads_shelves import get_shelves_given_titles
from preprocessor import PreProcessor
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
    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, 0)
    dictionary, corpus, books = preprocessor.get_library_info()
    lda = Library(dictionary, corpus, num_topics = 50, chunksize=1000, model_type = 'LDA')
    for title in tqdm(books):
         lda.add_book(books[title])
         break

    topics = lda.get_topics('text')

    for book in lda.books:
        print(book.book_title)
        print(book.get_topics())

if __name__ == '__main__':
    main()
