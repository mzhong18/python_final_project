"""

This program works out the original IDF count for each word in the NLTK provided Reuter's Corpus
Please simply run the file and only do it once.

Team: Jade Garisch, Marjan Kamyab, Micheal Zhong
Last update: December 19, 2018


"""

import nltk
import math
import string
import re
import numpy as np

files = [nltk.corpus.reuters.words(name) for name in nltk.corpus.reuters.fileids()]
file_names = [name for name in nltk.corpus.reuters.fileids()]

#this function creates a dictionary with the IDF
def get_IDF(corpus):
    lemma = nltk.stem.WordNetLemmatizer()
    words = set(
        [lemma.lemmatize(word.lower()) for word in set(nltk.corpus.reuters.words()) if word not in string.punctuation])
    idf_dict = dict.fromkeys(words, 0)
    c = []
    for thing in corpus:
        c.append(set([lemma.lemmatize(word.lower()) for word in thing if word not in string.punctuation]))
    # kept progress updates as they are helpful.
    m = 0
    print('Amount of words searched: ')
    for word in words:
        for thing in c:
            if word in thing:
                idf_dict[word] += 1
            m += 1
            if m % 10000000 == 0:
                print(m)
    return idf_dict

#this function returns the amount of articles in the entire corpus
def corpus_size(corpus):
    return len(corpus)

#this function saves the corpus size to a file
def save_size(size):
    with open('size.txt', 'w') as f:
        f.write('%d' % size)

#function saves a data structure to a numpy file
def numpy_it(filename, thing):
    np.save(filename, thing)

#main method
if __name__ == "__main__":
    print("Searching through corpus and creating IDF dictionary... Please wait until you see TRAINING DONE")
    idfs = get_IDF(files)
    print("TRAINING DONE")
    numpy_it('idf_dict.npy', idfs)
    size = corpus_size(files)
    save_size(size)
    numpy_it('files.npy', file_names)
