import operator, re, random, sys   
import numpy as np
from nltk.corpus import reuters, stopwords
from random import randint

'''
train_docs = list(filter(lambda doc: doc.startswith("train"), reuters.fileids()))
test_docs = list(filter(lambda doc: doc.startswith("test"), reuters.fileids()))
categories = reuters.categories()
'''

class Document:
    '''A class for a document from the Reuters data-set'''
    def __init__(self, category, index, m):
        self.m = m 
        self.category = category
        self.index = index
        self.id = reuters.fileids(category)[index]
        self.words = reuters.words(self.id)
        self.clean_data = self.remove_stops()
        self.features = set()
        for i in range(len(self.clean_data)-self.m+1):
            self.features.add(self.clean_data[i:i+self.m])
        tuples = {}
        for f in self.features:
            tuples[f] = self.clean_data.count(f)
        tuples_sorted = sorted(tuples, key=tuples.get, reverse=True)
        self.freq_features = tuples_sorted
        self.freq_features_counts = [tuples[tuples_sorted[i]] for i in range(len(tuples_sorted))]

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
    
    def get_top_features(self):
        return self.freq_features[:self.m]

    def get_top_features_counts(self):
        return self.freq_features_counts[:self.m]

    def __repr__(self):
        return 'category: ' + self.category + '\n'\
        + 'index: ' + str(self.index) + '\n'\
        +'----------------------------\n RAW DOCUMENT:'\


class SSK:
    def __init__(self, cat_a, cat_b, cat_a_training_count, cat_a_testing_count, cat_b_training_count, cat_b_testing_count, max_features, k, lamda, seed=None):
        '''
            catA: A category index for the Reuters data-set
            catB: A category index for the Reuters data-set
            k: the length of features
        '''
        self.cat_a = cat_a
        self.cat_b = cat_b
        self.cat_a_count = len(reuters.fileids(cat_a))
        self.cat_b_count = len(reuters.fileids(cat_b))
        self.cat_a_training_count = cat_a_training_count
        self.cat_a_testing_count = cat_a_testing_count
        self.cat_b_training_count = cat_b_training_count
        self.cat_b_testing_count = cat_b_testing_count
        if max(self.cat_a_training_count+self.cat_a_testing_count, self.cat_b_training_count+self.cat_b_testing_count) > min(self.cat_a_count, self.cat_b_count):
             print('number of trainig/testing documents exceeds number of articles')
             sys.exit(0)
        self.k = k
        self.lamda = lamda
        self.max_features = max_features
        self.kernel_matrix = np.zeros([cat_a_training_count+cat_b_training_count,cat_a_training_count+cat_b_training_count])
        self.top_feature_list = set()
        self.count_of_occurances = []
        self.seed = seed
        self.testing_list = []
        self.training_list = []
    
    def set_matrix(self):
        '''Create the matrix here'''
        # create list of lists where each inner-list is [1/-1,index]
        self.testing_list = [[self.cat_a,i,-1] for i in range(self.cat_a_training_count)]+[[self.cat_b,i,1] for i in range(self.cat_b_training_count)]
        random.shuffle(self.testing_list)

        for doc in self.testing_list:
            if(doc[0]==self.cat_a):
                doc_obj = Document(self.cat_a, doc[1], self.k)
            else:
                doc_obj = Document(self.cat_b, doc[1], self.k)
            
            for feature in doc_obj.get_top_features():
               self.top_feature_list.add(feature)

        for i in range(len(self.testing_list)):
            for j in range(i, len(self.testing_list)):
                self.kernel_matrix[i,j] = self.calc_kernel(self.testing_list[i], self.testing_list[j])*self.testing_list[i][2]*self.testing_list[j][2]
                self.kernel_matrix[j,i] = self.kernel_matrix[i,j]

    def calc_kernel(self, doc_1, doc_2):
        doc_1_words = Document(doc_1[0], doc_1[1], self.k).words
        doc_2_words = Document(doc_2[0], doc_2[1], self.k).words
        total = 0
        for feature in self.top_feature_list:
            l = doc_1_words.count(feature)
            j = doc_2_words.count(feature)
            total+= l*j*self.lamda**(2 * self.k)
        return total

ss = SSK("earn","corn", 2, 2, 2, 2, 30, 3, 0.8)
ss.set_matrix()
print(ss.kernel_matrix)