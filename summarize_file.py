
"""

This program summarizes .txt files - we have provided a few examples

Team: Jade Garisch, Marjan Kamyab, Micheal Zhong
Last update: December 19, 2018

INSTRUCTIONS: We have already put some code to run one of our examples 'file = Summarize_file("test.txt")'. 
In order to try our summarizer on other .txt files please change file = Summarize_file("test.txt") to 
file = Summarize_file(yourfilename) and run the file. The IDF dictionary and its data will update and grow everytime 
you summarize a new file in summarize.py. Everytime you run a file, you can view the results in results.txt



"""
import nltk
import math
import string
import re
import numpy as np
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer

class Summarize_file:
    
    def __init__(self, filename):
        with open(filename, 'r') as myfile:
            self.string = myfile.read()
            self.word_tokens = nltk.word_tokenize(self.string)
            self.sent_tokens = nltk.sent_tokenize(self.string)
            self.filename = filename
        # load saved IDF dictionary object
        self.idfs = np.load('idf_dict.npy').item()
        self.files = np.load('files.npy')
        with open("size.txt", "r") as f:
            integers = []
            for val in f.read().split():
                integers.append(int(val))
                self.size = integers[0]
        self.word_types =   list(set([nltk.stem.WordNetLemmatizer().lemmatize(word.lower()) for word in
                            self.word_tokens if word not in string.punctuation and word not in
                            nltk.corpus.stopwords.words('english')]))

    # Here, pass in the list of words in a document to grab the document TF score.
    # For testing purposes, pass in from the files variable. e.g. files[0]
    def get_TF(self):
        tf_dict = {}
        lemma = nltk.stem.WordNetLemmatizer()
        lemma_list = list(set([lemma.lemmatize(word.lower()) for word in self.word_tokens if word not in string.punctuation]))
        lemma_dict = {word: lemma_list.count(word) for word in lemma_list}
        maximum = max(lemma_dict.values())
        for word in lemma_list:
            tf_dict[word] = 0.5 + 0.5 * (lemma_list.count(word) / float(len(lemma_list))) / maximum
        return tf_dict

    #updating IDF
    def idf_update(self):
        lemma = nltk.stem.WordNetLemmatizer()
        lemmas = set([lemma.lemmatize(word.lower()) for word in set(self.word_tokens) if word not in string.punctuation])
        for thing in lemmas:
            if thing not in self.idfs:
                self.idfs[thing] = 1
            else:
                self.idfs[thing] += 1
        np.save('idf_dict.npy', self.idfs)
        return self.idfs #remember to save after updating

    def files_update(self):
        self.files = np.append(self.files, self.filename)
        np.save('files.npy', self.files)
        return self.files

    def size_update(self):
        self.size += 1
        with open('size.txt', 'w') as f:
            f.write(str(self.size))
        return self.size

    # This is fairly self-explanatory.
    def compute_TFIDF(self, tfs, idfs):
        tf_idf = {}
        for word, value in tfs.items():
            # Skip new article processing relic.
            if word == ' ':
                continue
            tf_idf[word] = value * idfs[word]
        return tf_idf

    # WIP. Pass in a sentence tokenized file and indicate how many words and sentences one would like to grab.
    def document_keywords(self, threshold):
        # The very first if statement has not been fully tested yet.
        # np.nditer(a)
        if self.filename not in np.nditer(self.files):
            self.idfs = self.idf_update()
            self.size = self.size_update()
            self.files = self.files_update()
        ids = range(0, len(self.sent_tokens))
        temp = []
        ridfs = {}
        # +1 to increase the overall document count. WIP
        for word, value in self.idfs.items():
            ridfs[word] = math.log10(self.size / float(value))
        for thing in self.sent_tokens:
            temp.extend(thing)
        tf = self.get_TF()
        tf_idf = self.compute_TFIDF(tf, ridfs)
        ranked = sorted(tf_idf.items(), reverse=True, key=lambda v: v[1])
        keywords = ranked[:threshold]
        sent_dict = dict.fromkeys(ids, 0)
        # Assume that longer sentences are more informative.
        for id in ids:
            word_list = self.sent_tokens[id]
            for word in word_list:
                for keyword in keywords:
                    if re.match(keyword[0], word.lower()):
                        sent_dict[id] += 1
            sent_dict[id] *= math.log2(len(self.sent_tokens[id]))
        summary = sorted(sent_dict.items(), reverse=True, key=lambda v: v[1])
        sents = summary[:threshold]
        return [self.sent_tokens[i] for (i, n) in sorted(sents, key=lambda v: v[0])]

    # This function takes the summary of the text as its parameter and measures the ratio of
    # the word types in the summary to the types in the original text
    def content_fraction(self, summ):
        stop = nltk.corpus.stopwords.words('english')
        lemma = nltk.stem.WordNetLemmatizer()
        summ_tokens = nltk.word_tokenize(' '.join(summ))
        summ_types = set([lemma.lemmatize(word.lower()) for word in summ_tokens if
                        word not in string.punctuation and word not in stop])
        return '%.3g' % (len(summ_types) / len(self.word_types) * 100)

    # This function takes the summary of the text as its parameter and measures how
    # much of the lexical diversity is lost compared to the original text
    def diversity_loss(self, summ):
        stop = nltk.corpus.stopwords.words('english')
        lemma = nltk.stem.WordNetLemmatizer()
        original_diversity = (len(self.word_types) / len(self.word_tokens))
        summ_tokens = nltk.word_tokenize(' '.join(summ))
        summ_types = set([lemma.lemmatize(word.lower()) for word in summ_tokens if
                            word not in string.punctuation and word not in stop])
        summ_diversity = (len(summ_types) / len(summ_tokens))
        return '%3g' % (original_diversity - summ_diversity)

    # This function takes the summary of the text as its parameter and measures the similarity
    # between the original text and the summary
    def cosine_sim(self, summ):
        vectorizer = TfidfVectorizer(stop_words='english')
        lemma = nltk.stem.WordNetLemmatizer()
        summ_string = ' '.join(summ)
        vectors = vectorizer.fit_transform([self.string, summ_string])
        return '%3g' % ((vectors * vectors.T).A)[0, 1]

    def print_summ(self, summary):
        new_s = ''
        for sent in summary:
            new_s = new_s + sent + " "
        formatted = textwrap.fill(str(new_s), 100)
        original = textwrap.fill(str(self.string), 100)
        with open('results.txt', 'w') as f:
            print('SUMMARY OF ' + self.filename.upper() + ':', file=f)
            print(formatted, file=f)
            print('\n', file=f)
            print('ANALYSIS:', file=f)
            print("Content Fraction = " + str(self.content_fraction(summ)) + "%", file=f)
            print("Summary Similarity = " + self.cosine_sim(summ), file=f)
            print("Lexical Diversity Reduction: " + str(self.diversity_loss(summ)), file=f)
            print('\n', file=f)
            print('ORIGINAL FILE:', file=f)
            print(original, file=f)


if __name__ == '__main__':

    #summarize a different file by changing the filename below from 'test.txt' to <yourTextFile>
    file = Summarize_file("test.txt")
    summ = file.document_keywords(6)
    file.print_summ(summ)
    print('Please see results and analysis in results.txt')

    # MORE THINGS THAT YOU CAN RUN
    # file = Summarize_file("html_file.txt")
    # summ = file.document_keywords(6)
    # file.print_summ(summ)
    # print('Please see results and analysis in results.txt')

    # AND ANOTHER
    # file = Summarize_file("test2.txt")
    # summ = file.document_keywords(6)
    # file.print_summ(summ)
    # print('Please see results and analysis in results.txt')
