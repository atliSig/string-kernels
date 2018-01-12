import operator, re
from nltk.corpus import reuters, stopwords
from random import randint 

'''
train_docs = list(filter(lambda doc: doc.startswith("train"), reuters.fileids()))
test_docs = list(filter(lambda doc: doc.startswith("test"), reuters.fileids()))
categories = reuters.categories()
'''

class Kernel:
    '''A SSK kernel class'''
    def __init__(self, n, m, document):
        '''
            n: the length of features
            m: the number of features used in the Kernel
            document: an object of type Document
        '''
        self.n = n
        self.m = m
        self.document = document
        self.document.set_features(n)
        self.document.set_freq_features()
        self.set_selected_features()
    def __repr__(self):
        print('A kernel with n: '+ str(self.n))

    def set_selected_features(self):
        self.selected_features = self.document.get_top_features(self.m)
        self.selected_features_counts = self.document.get_top_features_counts(self.m)

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
        return ' '.join([s for s in self.words if not re.match(r"[.,:;_\-&%<>!?=]",s) and s.lower() not in stopwords.words('english')])

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
        self.freq_features_counts = [tuples[tuples_sorted[i]] for i in range(len(tuples_sorted))]
    
    def get_top_features(self, m):
        return self.freq_features[:m]

    def get_top_features_counts(self, m):
        return self.freq_features_counts[:m]

    def __repr__(self):
        return 'category: ' + self.category + '\n'\
        + 'index: ' + str(self.index) + '\n'\
        +'----------------------------\n RAW DOCUMENT:'\

# example of use
ker = Kernel(4, 10, Document("earn", 0))
