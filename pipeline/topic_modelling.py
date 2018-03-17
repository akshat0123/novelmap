class TopicModelling:

    def __init__(self, dictionary, corpus, tf_idf_model, lsi_model, lda_model, num_topics, num_passes):
        self.dictionary = dictionary
        self.corpus = corpus
        self.lsi_model = lsi_model
        self.num_topics = num_topics
        self.num_passes = num_passes

    def generate_tf_idf_model(self):
        tfidf_model = tf_idf_model(self.corpus, id2word=self.dictionary)
        return tfidf_model

    def generate_lsi_model(self):
        tfidf_model = self.generate_tf_idf_model()
        model = lsi_model(tfidf_model[self.corpus], id2word=self.dictionary, self.num_topics)
        return model

    def generate_lda_model(self):
        model = lda_model(self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=self.num_passes)
        return model
