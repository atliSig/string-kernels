'''An SSK lazy implementation'''
import re
import random
import sys
from math import sqrt
import numpy as np
from nltk.corpus import reuters, stopwords
from math import sqrt
from cvxopt.solvers import qp
from cvxopt import matrix

'''
train_docs = list(filter(lambda doc: doc.startswith("train"), reuters.fileids()))
test_docs = list(filter(lambda doc: doc.startswith("test"), reuters.fileids()))
categories = reuters.categories()
'''

class Document:
    '''A class for a document from the Reuters data-set'''
    def __init__(self, category, index, m, n):
        '''
            :param m: the number of top features
            :param n: the length of each feature
        '''
        self.m = m
        self.n = n
        self.category = category
        self.index = index
        self.id = reuters.fileids(category)[index]
        self.words = reuters.words(self.id)
        self.clean_data = self.remove_stops()
        self.features = set()
        for i in range(len(self.clean_data)-self.n+1):
            self.features.add(self.clean_data[i:i+self.n])
        tuples = {}
        for feature in self.features:
            tuples[feature] = self.clean_data.count(feature)
        tuples_sorted = sorted(tuples, key=tuples.get, reverse=True)
        self.freq_features = tuples_sorted
        self.freq_features_counts = [tuples[tuples_sorted[i]] for i in range(len(tuples_sorted))]

    def get_words(self):
        '''Returns the original version of the Reuters document'''
        return reuters.words

    def remove_stops(self):
        '''
            Removes stopwords and low-case-ifies the document into
            into a list, split on spaces
        '''
        return ' '.join([s.lower() for s in self.words if not
            re.match(r"[.,:;_\-&%<>!?=]", s) and s.lower() not
            in stopwords.words('english')])

    def set_features(self):
        '''Sets the complete list of contigous letter combinations of length n'''
        self.features = set()
        for i in range(len(self.clean_data)-self.n+1):
            self.features.add(self.clean_data[i:i+self.n])

    def set_freq_features(self):
        '''returns features in the order of number of occurrences'''
        tuples = {}
        for f in self.features:
            tuples[f] = self.clean_data.count(f)
        tuples_sorted = sorted(tuples, key=tuples.get, reverse=True)
        self.freq_features = tuples_sorted
        self.freq_features_counts = [tuples[tuples_sorted[i]] for i in range(len(tuples_sorted))]

    def get_top_features(self):
        '''Returns the list of top features for this Document'''
        return self.freq_features[:self.m]

    def get_top_features_counts(self):
        '''Returns the number of occurrences for each top feature'''
        return self.freq_features_counts[:self.m]

    def __repr__(self):
        return 'category: ' + self.category + '\n'\
        + 'index: ' + str(self.index) + '\n'\
        +'----------------------------\n RAW DOCUMENT:'\

class SSK:
    '''A class for a lazy SSK implementation'''
    def __init__(self, cat_a, cat_b, cat_a_tr_c, cat_a_tst_c,
        cat_b_tr_c, cat_b_tst_c, max_features, k, lamda, seed=None):
        '''
            :param cat_a: A category index for the Reuters data-set
            :param cat_b: A category index for the Reuters data-set
            :param k: the length of features
            :param cat_a_tr_count: number of training samples from cat_a
            :param cat_a_tst_count: number of testing samples from cat_a
            :param cat_b_tr_count: number of training samples from cat_b
            :param cat_b_tst_count: number of testing samples from cat_b
            :param max_features: the number of features selected for each document
            :param k: the length of each feature
            :param lamda: the value of the distance constraint parameter
            :param seed: optional random seed
        '''
        self.cat_a = cat_a
        self.cat_b = cat_b
        self.cat_a_count = len(reuters.fileids(cat_a))
        self.cat_b_count = len(reuters.fileids(cat_b))
        self.cat_a_tr_c = cat_a_tr_c
        self.cat_a_tst_c = cat_a_tst_c
        self.cat_b_tr_c = cat_b_tr_c
        self.cat_b_tst_c = cat_b_tst_c
        self.k = k
        self.lamda = lamda
        self.max_features = max_features
        self.kernel_matrix = np.zeros([cat_a_tr_c+cat_b_tr_c, cat_a_tr_c+cat_b_tr_c])
        self.top_feature_list = set()
        self.count_of_occurances = []
        self.seed = seed
        self.training_list = []
        self.testing_list = []

        if cat_a_tr_c+cat_a_tst_c > self.cat_a_count or \
            cat_b_tr_c+cat_b_tst_c > self.cat_b_count:
             print('number of trainig/testing documents exceeds number of articles')
             sys.exit(0)
        if lamda <= 0 or lamda > 1:
            print('lamda must be in ]0,1]')
            sys.exit(0)
    
    def set_matrix(self):
        '''Create the matrix here'''
        # create list of lists where each inner-list is [1/-1,index]
        self.training_list = [[self.cat_a,i,-1] for i in
            range(self.cat_a_tr_c)]+[[self.cat_b,i,1] for i in range(self.cat_b_tr_c)]
        random.shuffle(self.training_list)

        # create the testing list
        self.testing_list =\
            [[self.cat_a, i, -1] for i in
                range(self.cat_a_tr_c+1, self.cat_a_tr_c + self.cat_a_tst_c+1)] +\
            [[self.cat_b, i, 1] for i in
                range(self.cat_b_tr_c+1, self.cat_b_tr_c + self.cat_b_tst_c+1)]
        random.shuffle(self.testing_list)

        for doc in self.testing_list:
            if doc[0] == self.cat_a:
                doc_obj = Document(self.cat_a, doc[1], self.max_features, self.k)
            else:
                doc_obj = Document(self.cat_b, doc[1], self.max_features, self.k)

            for feature in doc_obj.get_top_features():
                self.top_feature_list.add(feature)

        self.testing_list = [[self.cat_a, i, -1] for i in
            range(self.cat_a_tr_c)]+[[self.cat_b, i, 1] for i in range(self.cat_b_tr_c)]
        random.shuffle(self.testing_list)

        for doc in self.testing_list:
            if doc[0]==self.cat_a:
                doc_obj = Document(self.cat_a, doc[1], self.max_features, self.k)
            else:
                doc_obj = Document(self.cat_b, doc[1], self.max_features, self.k)

            for feature in doc_obj.get_top_features():
                self.top_feature_list.add(feature)

        for i in range(len(self.testing_list)):
            for j in range(i, len(self.testing_list)):
                self.kernel_matrix[i, j] = self.calc_kernel(self.testing_list[i],
                    self.testing_list[j])*self.testing_list[i][2]*\
                    self.testing_list[j][2]
                self.kernel_matrix[j, i] = self.kernel_matrix[i, j]

        #Normalizing results in rank issues with cvxopt.qp
        #self.normalize_kernel()



    def calc_kernel(self, doc_1, doc_2):
        '''Calculates the kernel matrix value for K[i,j]'''
        doc_1_words = Document(doc_1[0], doc_1[1], self.max_features, self.k).clean_data
        doc_2_words = Document(doc_2[0], doc_2[1], self.max_features, self.k).clean_data
#        doc_1_words = Document(doc_1[0], doc_1[1], self.max_features, self.k).words
#        doc_2_words = Document(doc_2[0], doc_2[1], self.max_features, self.k).words

        total = 0
        for feature in self.top_feature_list:
            l = doc_1_words.count(feature)
            j = doc_2_words.count(feature)
            total += l*j*self.lamda**(2*self.k)
        return total

    def normalize_kernel(self):
        '''Frobenius-Normalizes the kernel'''
        for i in range(len(self.training_list)):
            for j in range(len(self.training_list)):
                self.kernel_matrix[i, j] = self.kernel_matrix[i, j]/\
                sqrt(self.kernel_matrix[i, i]*self.kernel_matrix[j, j])

    def predict(self):
        '''Based on the kernel, make predictions using cvxopt.qp'''
        G = -np.eye(len(self.training_list))
        G = np.append(G, np.eye(len(self.training_list)))
        G.resize(2 * len(self.training_list), len(self.training_list))

        C = 10
        h = np.zeros(len(self.training_list))
        h_alpha = np.ones(len(self.training_list)) * C

        h = np.append(h, h_alpha)
        h.resize(2 * len(self.training_list))
        q = -np.ones((len(self.training_list)))

        # Optimizes the alpha values
        r = qp(matrix(self.kernel_matrix), matrix(q), matrix(G), matrix(h))
        alpha = list(r['x'])

        # calculates the alphas that are larger than the threshold
        alpha_list = self.get_alpha(alpha, self.training_list, 10**-5)

        # predictions

        correctness = 0
        for case in self.testing_list:
            estimate = self.ind(case, alpha_list)
            if estimate > 0:
                if case[2]==1:
                    print("Correct")
                    correctness += 1
                else:
                    print("Wrong")
            elif estimate == 0:
                print("uncertain")
            else:
                if case[2] == -1:
                    print("Correct")
                    correctness += 1
                else:
                    print("Wrong")
        print(str(correctness/len(self.testing_list)))

    def get_alpha(self, alpha, data, threshold):
        '''Returns the list of alphas [HUGO]'''
        return [[data[idx], al_el] for idx,al_el in enumerate(alpha)]

    def ind(self, x, alpha_list):
        '''[HUGO]'''
        return np.sum([a[1] * a[0][2] * self.calc_kernel(a[0], x) for a in alpha_list])

    def print_kernel(self):
        '''A more readable way of printing the kernel matrix'''
        np.set_printoptions(precision=3, suppress=True)
        print(self.kernel_matrix)

    def __repr__(self):
        return "i'm an SSK!"


if __name__ == '__main__':
    if len(sys.argv) != 10:
        print(len(sys.argv))
        print('Incorrect number of input arguments!')
        sys.exit(1)
    else:
        ssk = SSK(sys.argv[1], sys.argv[2], int(sys.argv[3]),
            int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
            int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9]))
        ssk.set_matrix()
        ssk.predict()
        ssk.print_kernel()
