import urllib.request, json, time, os
from bs4 import BeautifulSoup
from tqdm import tqdm


# Good reads base url and API key
BASE_URL = 'https://www.goodreads.com/book/title.xml?';
GOODREADS_KEY = 'CQquiYWQqp8iiw10w38tYA'


def get_titles(path):
    """ Retrieves titles from list of titles at the path
    """

    titles = []
    with open(path, 'r') as f:
        for line in f.readlines():
            titles.append(line.rstrip('\n')) 

    return titles


def get_url(base, title, key):
    """ Generates a Goodreads API url given the base url, the title, and the
        goodreads key
    """

    return '%stitle=%s&key=%s' % (base, '+'.join(title.split(' ')), key)


def get_shelves_given_title(base, title, key):
    """ Takes in a base url, a book title, and the goodreads key and returns the
        list of shelves the book appears in
    """

    url = get_url(base, title, key)
    html = urllib.request.urlopen(url)
    soup = BeautifulSoup(html.read(), 'lxml');
    shelves = [shelf['name'] for shelf in soup.findAll('shelf')]

    return shelves

def get_shelves_given_titles(input_path, output_path):
    """ Takes in a path for a file with a list of titles and the output path and
        writes the titles and their corresponding shelves to the specified output
        path
    """

    titles = get_titles(input_path)
    book_genres = {}

    for title in tqdm(titles):
        shelves = get_shelves_given_title(BASE_URL, title, GOODREADS_KEY)
        book_genres[title] = shelves

        # Time delay for API access limit
        time.sleep(1)

    with open(output_path, 'w+') as f:
        f.write(json.dumps(book_genres))
