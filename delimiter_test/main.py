import re

BOOK_PATH = '../data/raw/The Idiot.txt'

def main():

    section_splitter = '\s\s[IVXL][IVXL]*[IVXL]*[IVXL]*[IVXL]*[IVXL]*\s\s'
    end_splitter = 'THE END'

    book_string = ''
    with open(BOOK_PATH, 'r') as f:
        for line in f.readlines():
            book_string += line


    book_text = re.split(end_splitter, book_string)[0]
    chapters = re.split(section_splitter, book_text)[1:]

    print(len(chapters))


if __name__ == '__main__':
    main()
