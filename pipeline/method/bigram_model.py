from random import uniform 


class BigramModel:


    def __init__(self):

        self.unigram_term_freqs = {}
        self.bigram_term_freqs = {}
        self.unigram_total_term_freqs = 0
        self.bigram_total_term_freqs = {}


    def add_unigram(self, unigram):
        """ Adds a unigram to the unigram frequency dictionary
        """

        if unigram in self.unigram_term_freqs:
            self.unigram_term_freqs[unigram] += 1
        else:
            self.unigram_term_freqs[unigram] = 1

        self.unigram_total_term_freqs += 1


    def add_bigram(self, first, second):
        """ Adds a bigram pair to the bigram frequency dictionary
        """

        if first in self.bigram_term_freqs:
            if second in self.bigram_term_freqs[first]:
                self.bigram_term_freqs[first][second] += 1
            else:
                self.bigram_term_freqs[first][second] = 1

            self.bigram_total_term_freqs[first] += 1

        else:
            self.bigram_term_freqs[first] = { second: 1 }
            self.bigram_total_term_freqs[first] = 1

    
    def add_document(self, document):
        """ Takes in a document as a list of tokens and updates the frequency
            dictionaries
        """

        # Add first word in document
        self.add_unigram(document[0]) 

        # Add remaining words in document
        for i in range(1, len(document)):

            self.add_unigram(document[i])
            self.add_bigram(document[i-1], document[i])


    def gen_term_probs(self, term_freqs, total_term_count):
        """ Takes in a dictionary of term frequencies and returns a dictionary
            of terms and their corresponding probabilities
        """
        term_probs = {}

        for term in term_freqs:
            term_probs[term] = term_freqs[term] / total_term_count

        return term_probs


    def gen_inverted_probs(self, term_probs):
        """ Takes in a dictionary of terms and their corresponding probabilities and
            returns a dictionary with keys in the interval between 0 and the total
            probability with corresponding terms as values, as well as the total
            probability amount for the term probability dictionary 
        """

        total_prob = 0
        inverted_probs = {}

        for term in term_probs:
            total_prob += term_probs[term]
            inverted_probs[total_prob] = term

        return inverted_probs, total_prob


    def gen_token(self, inverted_probabilities, total_probability):
        """ Takes in a dictionary with probability ranges mapped to corresponding
            terms and the total probability and returns a token based upon its
            probability of being generated
        """

        random_key_init = uniform(0, total_probability)
        random_key = min(inverted_probabilities.items(), key=lambda x: abs(random_key_init - x[0]))[0]

        token = inverted_probabilities[random_key]

        return token


    def gen_document(self, size):
        """ Takes in a size argument and returns a document containing that many
            tokens
        """

        unigram_term_probs = self.gen_term_probs(self.unigram_term_freqs, self.unigram_total_term_freqs)
        inverted_unigram_probs, total_unigram_prob = self.gen_inverted_probs(unigram_term_probs)

        current_term = self.gen_token(inverted_unigram_probs, total_unigram_prob)
        document = [current_term]

        for i in range(size - 1):

            if current_term in self.bigram_term_freqs:
                bigram_term_probs = self.gen_term_probs(self.bigram_term_freqs[current_term], self.bigram_total_term_freqs[current_term])
                inverted_bigram_probs, total_bigram_prob = self.gen_inverted_probs(bigram_term_probs)
                current_term = self.gen_token(inverted_bigram_probs, total_bigram_prob)

            else:
                current_term = self.gen_token(inverted_unigram_probs, total_unigram_prob)

            document.append(current_term)

        return document