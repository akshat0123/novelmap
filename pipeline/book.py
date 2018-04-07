

class Book:


    def __init__(self, dictionary, book):
        """ Takes in library dictionary, a dictionary containing a book title, a
            list of tokens for the entire book and a list containing lists of
            tokens for each chapter in the book
        """
        self.book_title = book['title']
        self.book_corpus = [dictionary.doc2bow(chapter_tokens)
                             for chapter_tokens in book['book_tokens']]
        self.chapter_corpi = [
            dictionary.doc2bow(
                book['chapter_tokens'][i]) for i in range(len(book['chapter_tokens'])
           )
        ]
        self.book_model = None

    def add_model(self, model):
        self.book_model = model

    def get_topics(self):
        return self.book_model.print_topics()