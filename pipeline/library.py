from book import Book


class Library:

    
    def __init__(self, dictionary, corpus):
        """ Takes in a gensim dictionary and corpus
        """
        self.dictionary = dictionary
        self.corpus = corpus
        self.books = []


    def add_book(self, book):
        """ Take dictionary containing title, list of tokens for the book, and a
            list containing lists of tokens for each chapter in the book
        """
        book = Book(self.dictionary, book)
        self.books.append(book)
