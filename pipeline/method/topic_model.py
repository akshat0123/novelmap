from bigram_model import BigramModel
from tqdm import trange
from tqdm import tqdm
import numpy as np


class TopicModel:


    def __init__(self, y_axis_count, y_axis_length, threshold):
        self.bigram_model = BigramModel(threshold)
        self.y_axis_count = y_axis_count
        self.y_axis_length = y_axis_length


    def add_document(self, document):
        """ Takes in a document as a list of tokens and updates the bigram model 
        """
        self.bigram_model.add_document(document)


    def add_documents(self, documents):
        """ Takes in a list of documents and updates the bigram model
        """
        for document in tqdm(documents, desc='Adding Documents to Model'):
            self.add_document(document)

    
    def gen_y_axis(self):
        """ Uses bigram model to generate y axis of specified size
        """
        return self.bigram_model.gen_document(self.y_axis_length)


    def gen_y_axes(self):
        """ Uses bigram model to generate specified number of y axes specified
            number of times
        """
        y_axes = []
        for i in trange(self.y_axis_count, desc='Generating Y Axes'):
            y_axes.append(self.gen_y_axis())

        return y_axes


    def gen_y_axes_feature_spaces(self, y_axes):
        """ Takes in a list of y axes generated from the 'generate_y_axes'
            function and returns a dictionary with tokens mapped tuples of the y
            axis list index in the given list, and the corresponding token id
            for the token key
        """
        token_dict = {}
        for i in range(len(y_axes)):
            y_axis = y_axes[i]

            for j in range(len(y_axis)):
                token = y_axis[j]

                if token in token_dict:
                    token_dict[token].append((i, j))

                else:
                    token_dict[token] = [(i, j)]

        return token_dict


    def map_doc_to_y_axes_feature_spaces(self, y_axes, doc):
        """ Takes in a document and a list of y axes generated from the
            'generate_y_axes' function and returns a mapping of the document in
            the feature space for each y axis
        """
        doc_mappings = { i: [] for i in range(len(y_axes)) }
        token_dict = self.gen_y_axes_feature_spaces(y_axes)
        for token in doc:
            if token in token_dict:
                for pair in token_dict[token]:
                    y_axis_num, token_label = pair[0], pair[1]
                    doc_mappings[y_axis_num].append(token_label)

        return doc_mappings


    def calc_real_fake_dists(self, document, y_axes):
        """ Takes in a document and a set of y axes generated from the
            'generate_y_axes' function and returns a list of average distances
            from the book to each unigram on each y axis
        """
        doc_mappings = self.map_doc_to_y_axes_feature_spaces(y_axes, document)

        # Compute distances averaged across the book length for the book to each fake book
        avg_dists = []
        for i in range(len(y_axes)):
            y_axis = y_axes[i]

            reals = np.array([doc_mappings[i] for j in range(len(y_axis))])

            # Map document to y axis feature space for the current y axis
            if reals.shape[1] != 0:

                # Generate fake documents for each unigram on the current y axis
                fakes = np.array([[j for k in range(len(doc_mappings[i]))] for j in range(len(y_axis))]) 

                # Calculate the distances between the real document and each
                # fake book for the current y axis
                dists = np.absolute(fakes - reals)

                # Calculate the average distances between the book and each
                # unigram for the current y axis
                avg_dists.append(np.average(dists, axis=1))

            else:
                avg_dists.append([self.y_axis_length for i in range(len(y_axis))])

        return avg_dists


    def calc_topics(self, documents, y_axes):
        """ Takes in a list of documents and a set of y axes generated from the
            'generate_y_axes' function and returns a list of average distances
            from all books to each unigram on each y axis
        """

        # Calculate average distances for each book and each y axis
        avg_dists = []
        for document in tqdm(documents, desc='Calculating Average Distances'):
            avg_dists.append(self.calc_real_fake_dists(document, y_axes))

        # Extract topics from all documents
        topic_freqs, doc_topics = {}, [{} for i in range(len(documents))] 
        for y_axis_index in trange(len(y_axes), desc='Extracting Topics'):
            y_axis = y_axes[y_axis_index]

            for doc_index in range(len(documents)):
                document = documents[doc_index]
                y_index = np.argmin(avg_dists[doc_index][y_axis_index])
                topic = y_axis[y_index]

                if topic in topic_freqs: 
                    topic_freqs[topic] += 1

                else: 
                    topic_freqs[topic] = 1

                if topic in doc_topics[doc_index]:
                    doc_topics[doc_index][topic].append(avg_dists[doc_index][y_axis_index])

                else:
                    doc_topics[doc_index][topic] = [avg_dists[doc_index][y_axis_index]]

        return topic_freqs, doc_topics
