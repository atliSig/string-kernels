import operator, re, random, sys   
import numpy as np
from nltk.corpus import reuters, stopwords
from random import randint

'''
train_docs = list(filter(lambda doc: doc.startswith("train"), reuters.fileids()))
test_docs = list(filter(lambda doc: doc.startswith("test"), reuters.fileids()))
categories = reuters.categories()
'''

class SSK:
    def __init__(self, cat_a, cat_b, m, n, max_features, k, lamda, seed=None):
        '''
            catA: A category index for the Reuters data-set
            catB: A category index for the Reuters data-set
            m: The number of testing documents from catA and catB
            n: The number of training document
            k: the length of features
        '''
        self.m = m
        self.n = n
        self.cat_a = cat_a
        self.cat_b = cat_b
        self.cat_a_count = len(reuters.fileids(cat_a))
        self.cat_b_count = len(reuters.fileids(cat_b))
        if self.m+self.n > min(self.cat_a_count, self.cat_b_count):
             print('number of trainig/testing documents exceeds number of articles')
             sys.exit(0)
        self.k = k
        self.lamda = lamda
        self.max_features = max_features
        self.matrix = np.zeros([])
        self.training_docs = []
        self.testing_docs = []
        self.top_feature_list = set()
        self.count_of_occurances = []
        self.seed = seed
        self.set_docs()


    def set_docs(self):
        '''A naive random split into training/testing'''
        index = []
        if(self.seed):
            random.seed(self.seed)
            index = sorted(range(self.m+self.n), key=lambda *args: random.random())
        else:
            index = sorted(range(self.m+self.n), key=lambda *args: random.random())
        self.training_docs = index[self.n:]
        self.testing_docs = index[:self.n]

    def set_matrix(self):
        '''Create the matrix here'''
        for doc in self.testing_docs:
            curr_doc = Document(self.cat_a, doc)
            curr_doc.set_features(self.k)
            curr_doc.set_freq_features()
            top_features = curr_doc.get_top_features(self.max_features)
           # top_features_count = [(curr_doc.get_top_features_counts(self.max_features))]
            top_feature_list.add(top_features)
            #count_of_occurances.add(top_features_count)
            curr_doc = Document(self.cat_b, doc)
            curr_doc.set_features(self.k)
            curr_doc.set_freq_features()
            top_features = curr_doc.get_top_features(self.max_features)
           # top_features_count = [(curr_doc.get_top_features_counts(self.max_features))]
            top_feature_list.add(top_features)
            #count_of_occurances.add(top_features_count)
            #top_features_count = curr_doc.get_top_features_counts(self.max_features)
        # Create kernel matrix
        # N * N matrix from kernel documents
        # k = length of subsequence
        kernel = np.zeros((self.n,self.n))
        for n in range(len(self.testing_docs)):
            for i in range(n,len(self.testing_docs)):
                for m in range(len(top_features_list))
                l = Document(self.cat_a, )
                j =Document(self.cat_b,)
                if j.contains(m).count
                if l.contains(m).count
                kernel[][] = j*lamda**(4)+l*lamda**(4)
                kernel[n][i] = 

       # for i in range(n):
       #     for j in range(n):
       #         sum = 0
       #         for m in range(m*m):
                    ### Dot product to create the kernel ###
       #             sum += (count_of_occurances_lamda[j][i]*count_of_occurances_lamda[][])
        #            kernel[j][i]




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
        '''Sets the complete list of contigous letter combinations of length n'''
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

if __name__ == '__main__':
    # Call stuff from here