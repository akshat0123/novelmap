from gensim.models import LdaModel
from gensim.models import TfidfModel
from gensim.models import LsiModel
from book import Book


class Library:

    
    def __init__(self, dictionary, corpus, num_topics=50, chunksize=2, passes=1, model_type='LDA'):
        """ Takes in a gensim dictionary and corpus
        """
        self.dictionary = dictionary
        self.corpus = corpus
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.books = []
        self.model_type = model_type
        self.model = self.get_model_library()

    def get_model_library(self):
        
        if self.model_type == 'LDA':
            model = LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.dictionary,
                                         chunksize=self.chunksize, passes=self.passes)
        return model

    def get_model_book(self, book_corpus):
        
        if self.model_type == 'LDA':
            model = LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.dictionary,
                                         chunksize=self.chunksize, passes=self.passes)
        return model
            
        '''elif model == 'LSI':
            tfidf_model = TfidfModel(self.corpus, id2word=self.dictionary)
            self.model = LsiModel(tfidf_model[self.corpus], num_topics=self.num_topics, id2word=self.dictionary, chunksize=2)

        elif model == 'TFIDF':
            self.model = TfidfModel(self.corpus, id2word=self.dictionary)'''


    def add_book(self, book):
        """ Take dictionary containing title, list of tokens for the book, and a
            list containing lists of tokens for each chapter in the book
        """
        book = Book(self.dictionary, book)
        #book_model = self.get_model_book(book.book_corpus)
        #book.add_model(book_model)
        self.books.append(book)

    def get_topics(self, book):
        """ Takes in a text and returns the topics for that book
        """
        return self.model.print_topics()


    def get_keywords(self, text):
        """ Takes in a text and returns the keywords for that book
        """
        pass


    def get_topic_vectors(self, text):
        """ Takes in a text and returns the topics for that book in vector
            format
        """
        pass


    def get_keyword_vectors(self, text):
        """ Takes in a text and returns the keywords for that book in vector
            format
        """
        pass


    def get_similarity(booka, bookb):
        """
        """
        pass
    

    def cluster(self, k):
        """ Gets topic vectors for all books and clusters them
        """
        pass
