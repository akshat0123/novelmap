import re

BOOK_PATH = '../data/raw/The Hound of the Baskervilles.txt'

def main():

    section_splitter = '\s\sChapter \d\d*\.'
    end_splitter = 'End of Project Gutenberg'

    book_string = ''
    with open(BOOK_PATH, 'r') as f:
        for line in f.readlines():
            book_string += line


    book_text = re.split(end_splitter, book_string)[0]
    chapters = re.split(section_splitter, book_text)[1:]

    print(len(chapters))


if __name__ == '__main__':
    main()
