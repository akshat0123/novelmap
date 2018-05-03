import re

BOOK_PATH = '../data/raw/Madame Bovary.txt'

def main():

    section_splitter = '\s\sChapter'
    end_splitter = 'End of the Project Gutenberg'

    book_string = ''
    with open(BOOK_PATH, 'r') as f:
        for line in f.readlines():
            book_string += line


    book_text = re.split(end_splitter, book_string)[0]
    chapters = re.split(section_splitter, book_text)[1:]

    print(len(chapters))


if __name__ == '__main__':
    main()
