from preprocessor import PreProcessor
from topic_model import TopicModel
from os.path import isfile
from filepaths import *
from tqdm import trange
from tqdm import tqdm
import numpy as np
import pickle



def main():

    # Get book texts split into chapters
    preprocessor = PreProcessor(DICT, CORP, DELIM, RAW, TOKEN, 0)
    dictionary, corpus, books = preprocessor.get_library_info()

    books = [books[book]['book_tokens'] for book in books]

    tm = TopicModel()
    for book in tqdm(books, desc='Adding Books to Model'):
        tm.add_document(book)


    y_axes = tm.gen_y_axes(100, 10)
    book_avg_dists = [] 
    for book in tqdm(books, desc='Calculating Topics'):
        book_avg_dists.append(tm.calc_real_fake_dists(book, y_axes))

    book_avg_dists = np.array(book_avg_dists)


if __name__ == '__main__':
    main()
