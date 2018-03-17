
class PreProcessing:
    #Generates gensim corpus from raw book text

    def __init__(self,tokenizer, lemmatizer, stop_words, book_directory, min_count):
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.book_directory = book_directory
        self.min_count = min_count

    # Raw text to list of tokens
    def tokenize(self, raw_text):
        return self.tokenizer.tokenize(raw_text)

    # List of tokens to list of stemmed tokens
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token)
                for token in tokens if token not in self.stop_words]

    def normalize_text(self, raw_text):
        translator = str.maketrans('', '', string.punctuation)
        raw_text = raw_text.translate(translator).lower()
        tokens = self.tokenize(raw_text)
        lemmatized_tokens = self.lemmatize(tokens)

        token_counts = Counter(lemmatized_tokens)
        normalized_tokens = [token for token in lemmatized_tokens
                             if token_counts[token] > MIN_COUNT]
        return normalized_tokens

    # Returns total corpus and dictionary in vector format
    def process_books(self):
        book_tokens = []
        for book in os.listdir(self.book_directory):
            with open(os.path.join(self.book_directory,book), 'r') as handle:
                book_content = handle.read()
                book_tokens.append(self.normalize_text(book_content))

        dictionary = corpora.Dictionary(book_tokens)
        dictionary.save(os.path.join(DICTIONARY_PATH)
        
        corpus = [dictionary.doc2bow(token) for token in book_tokens]
        corpora.MmCorpus.serialize(CORPUS_PATH, corpus)
        
        return dictionary, corpus
