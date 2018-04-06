from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from gensim import corpora
from os.path import isfile
import string, pickle, re


class Preprocessor:
    # Generates gensim corpus from raw book text


    def __init__(self, dict_path, corp_path, delim_path, raw_path, token_path, min_count):
        """ Takes in path to store/load dictionary and corpus, and the path for
            a file containing book titles and their corresponding chapter and
            ending delimiters, the file path containing all the raw book texts,
            and a min count value
        """
        self.lemmatizer = WordNetLemmatizer() 
        self.tokenizer = TreebankWordTokenizer()
        self.stop_words = set(stopwords.words('english'))
        self.dictionary_path = dict_path
        self.corpus_path = corp_path
        self.book_delimiter_path = delim_path
        self.book_raw_path = raw_path
        self.token_path = token_path
        self.min_count = min_count


    def tokenize(self, raw_text):
        """ Raw text to list of tokens
        """
        return self.tokenizer.tokenize(raw_text)


    def lemmatize(self, tokens):
        """ List of tokens to list of stemmed tokens
        """
        return [self.lemmatizer.lemmatize(token)
                for token in tokens if token not in self.stop_words]


    def normalize_text(self, raw_text):
        """ Tokenize, lemmatize, remove punctuation
        """
        translator = str.maketrans('', '', string.punctuation)
        raw_text = raw_text.translate(translator).lower()
        tokens = self.tokenize(raw_text)
        lemmatized_tokens = self.lemmatize(tokens)

        token_counts = Counter(lemmatized_tokens)

        normalized_tokens = [token for token in lemmatized_tokens if token_counts[token] > self.min_count]
        return normalized_tokens


    def remove_proper_nouns(self, text):
        """
        """
        pass


    def get_chapter_tokens(self, book_path, section_splitter, end_splitter):
        """ Takes in book file path, chapter splitting regex, post-content removal
            regex, and returns a list normalized tokens for each chapter in the book
        """
        with open(book_path) as f:
            book_text = ''
            for line in f.readlines():
                book_text += line

            book_text = re.split(end_splitter, book_text)[0]
            chapter_texts = re.split(section_splitter, book_text)[1:]
            chapter_texts = [self.normalize_text(text) for text in chapter_texts]

        return chapter_texts


    def get_book_delimiter_data(self):
        """ Uses book delimiter file and returns a dictionary with book titles
            mapped to the book's file path, chapter delimiter, and end delimiter
        """
        book_delimiter_data = {}
        with open(self.book_delimiter_path) as f:

            for line in f.readlines():
                title, section, end = line.rstrip('\n').split(',')

                if section != '' and end != '':
                    book_path = '%s/%s.txt' % (self.book_raw_path, title)

                    book_delimiter_data[title] = {
                        'path': book_path,
                        'section_delimiter': section,
                        'end_delimiter': end
                    }

        return book_delimiter_data


    def get_all_tokens(self):
        """ Reads in every book specified by the book delimiter data file and
            returns a list containing a list of all the tokens for each book and
            returns a dictionary of books with titles mapped to the tokens for
            the entire book as well as each chapter in the book
        """

        book_delimiter_data = self.get_book_delimiter_data()
        all_tokens, books = [], {}

        for title in book_delimiter_data:

            path = book_delimiter_data[title]['path']
            section = book_delimiter_data[title]['section_delimiter']
            end = book_delimiter_data[title]['end_delimiter']

            book_chapter_tokens = self.get_chapter_tokens(path, section, end)

            book_tokens = []
            for chapter_tokens in book_chapter_tokens:
                book_tokens += chapter_tokens
            
            books[title] = {
                'title': title,
                'book_tokens': book_tokens,
                'chapter_tokens': book_chapter_tokens
            }

            all_tokens.append(book_tokens)

        return all_tokens, books


    def process_books(self):
        """ Creates and returns a gensim dictionary and corpus as well as a
            dictionary of book titles mapped to the tokens for the entire book
            as well as each chapter in the book
        """

        all_tokens, books = self.get_all_tokens()

        dictionary = corpora.Dictionary(all_tokens)
        dictionary.save(self.dictionary_path)

        corpus = [dictionary.doc2bow(tokens) for tokens in all_tokens]
        corpora.MmCorpus.serialize(self.corpus_path, corpus)

        pickle.dump(books, open(self.token_path, 'wb'))

        return dictionary, corpus, books


    def get_library_info(self):
        """ Loads gensim dictionary and corpus if saved dictionary and corpus is
            available
            Else creates and returns a gensim dictionary and corpus as well as a
            dictionary of book titles mapped to the tokens for the entire book
            as well as each chapter in the book
        """

        if isfile(self.dictionary_path) and isfile(self.corpus_path) and isfile(self.token_path):

            dictionary = corpora.Dictionary.load(self.dictionary_path)
            corpus = corpora.MmCorpus(self.corpus_path)
            books = pickle.load(open(self.token_path, 'rb'))

        else:

            dictionary, corpus, books = self.process_books()

        return dictionary, corpus, books
