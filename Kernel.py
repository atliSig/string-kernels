from nltk.corpus import reuters, stopwords
import operator
from random import randint 

'''
train_docs = list(filter(lambda doc: doc.startswith("train"), reuters.fileids()))
test_docs = list(filter(lambda doc: doc.startswith("test"), reuters.fileids()))
categories = reuters.categories()
'''

class Kernel:
    '''A SSK kernel class'''
    def __init__(self, n, document):
        self.n = n
        self.document = document
        self.document.set_features(n)
        self.document.set_freq_features()
        print(self.document.freq_features)
    
    def __repr__(self):
        print('A kernel with n: '+ str(self.n))

class Document:
    '''A class for a document from the Reuters data-set'''
    def __init__(self, category, index):
        self.category = category
        self.index = index
        self.id = reuters.fileids(category)[index]
        self.words = reuters.words(self.id)
        self.clean_data = self.remove_stops()
    
    def get_words(self):
        return reuters.words
    
    def remove_stops(self):
        special_char = [".", ",", ":", "", "_", "-", "&", "%", "<", ">", "!", "?", "="]
        cleaned = [x.lower() for x in self.words if x not in special_char and x.lower() not in stopwords.words('english')]
        return ' '.join(cleaned)
    
    def set_features(self, n=4):
        '''returns the complete list of contigous letter combinations of length n'''
        self.features = set()
        for i in range(len(self.clean_data)-n+1):
            self.features.add(self.clean_data[i:i+n])

    def set_freq_features(self):
        '''returns features in the order of number of occurrences'''
        tuples = {}
        for f in self.features:
            tuples[f] = self.clean_data.count(f)
        tuples_sorted = sorted(tuples, key=tuples.get, reverse=True)
        self.freq_features = tuples_sorted

    def __repr__(self):
        return 'category: ' + self.category + '\n'\
        + 'index: ' + str(self.index) + '\n'\
        +'----------------------------\n RAW DOCUMENT:'\
        +reuters.raw(self.id)
