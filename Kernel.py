'''An SSK lazy implementation'''
import re
import random
import sys
import time
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
    def __init__(self, cat_a, cat_b, max_features, k, lamda, cat_a_tr_c=0, cat_a_tst_c=0, 
        cat_b_tr_c=0, cat_b_tst_c=0, run_test=True, seed=None):
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
        self.run_test = run_test
        if self.run_test:
            training_list_a = list(filter(lambda doc: doc.startswith("train"), self.cat_a))[:self.cat_a_tr_c]
            training_list_b = list(filter(lambda doc: doc.startswith("train"), self.cat_b))[:self.cat_a_tr_c]
            testing_list_a = list(filter(lambda doc: doc.startswith("test"), self.cat_a))[:self.cat_b_tst_c]
            testing_list_b = list(filter(lambda doc: doc.startswith("test"), self.cat_b))[:self.cat_b_tst_c]
            self.training_list = training_list_a + training_list_b
            self.testing_list = testing_list_a + testing_list_b
        else:
            self.training_list = []
            self.testing_list = []
        self.k = k
        self.lamda = lamda
        self.max_features = max_features
        self.kernel_matrix = np.zeros([cat_a_tr_c+cat_b_tr_c, cat_a_tr_c+cat_b_tr_c])
        self.top_feature_list = set()
        self.count_of_occurances = []
        self.seed = seed
        self.alpha_list_global = []
        if cat_a_tr_c+cat_a_tst_c > self.cat_a_count or \
            cat_b_tr_c+cat_b_tst_c > self.cat_b_count:
             print('number of training/testing documents exceeds number of articles')
             sys.exit(0)
        if lamda <= 0 or lamda > 1:
            print('lamda must be in ]0,1]')
            sys.exit(0)
    
    def set_matrix(self):
        '''Create the matrix here'''
        # create list of lists where each inner-list is [1/-1,index]
        if not self.run_test:
            self.training_list = [[self.cat_a,i,-1] for i in
                range(self.cat_a_tr_c)]+[[self.cat_b,i,1] for i in range(self.cat_b_tr_c)]
            # create the testing list
            self.testing_list =\
                [[self.cat_a, i, -1] for i in
                    range(self.cat_a_tr_c+1, self.cat_a_tr_c + self.cat_a_tst_c+1)] +\
                [[self.cat_b, i, 1] for i in
                    range(self.cat_b_tr_c+1, self.cat_b_tr_c + self.cat_b_tst_c+1)]
        
        random.shuffle(self.training_list)
        random.shuffle(self.testing_list)

        for doc in self.training_list:
            if doc[0] == self.cat_a:
                doc_obj = Document(self.cat_a, doc[1], self.max_features, self.k)
            else:
                doc_obj = Document(self.cat_b, doc[1], self.max_features, self.k)

            for feature in doc_obj.get_top_features():
                self.top_feature_list.add(feature)

        self.training_list = [[self.cat_a, i, -1] for i in
            range(self.cat_a_tr_c)]+[[self.cat_b, i, 1] for i in range(self.cat_b_tr_c)]
        random.shuffle(self.testing_list)

        for doc in self.training_list:
            if doc[0]==self.cat_a:
                doc_obj = Document(self.cat_a, doc[1], self.max_features, self.k)
            else:
                doc_obj = Document(self.cat_b, doc[1], self.max_features, self.k)

            for feature in doc_obj.get_top_features():
                self.top_feature_list.add(feature)

        for i in range(len(self.training_list)):
            for j in range(i, len(self.training_list)):
                self.kernel_matrix[i, j] = self.calc_kernel(self.training_list[i],
                    self.training_list[j])*self.training_list[i][2]*\
                    self.training_list[j][2]
                self.kernel_matrix[j, i] = self.kernel_matrix[i, j]

        #Normalizing results in rank issues with cvxopt.qp
        #self.normalize_kernel()

    def calc_kernel(self, doc_1, doc_2):
        '''Calculates the kernel matrix value for K[i,j]'''
        doc_1_words = Document(doc_1[0], doc_1[1], self.max_features, self.k).clean_data
        doc_2_words = Document(doc_2[0], doc_2[1], self.max_features, self.k).clean_data
        # doc_1_words = Document(doc_1[0], doc_1[1], self.max_features, self.k).words
        # doc_2_words = Document(doc_2[0], doc_2[1], self.max_features, self.k).words

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
        self.alpha_list_global = alpha_list

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

    def set_results(self, verbose=True):
        '''Print results for this Kernel'''
        # Class a is assigned positive values 
        a_tp = 0
        a_tn = 0
        a_fp = 0
        a_fn = 0
        # Class b is assigned negative values
        b_tp = 0
        b_tn = 0
        b_fp = 0
        b_fn = 0
        ###
        for case in self.testing_list:
            estimate = self.ind(case, self.alpha_list)
            #check for true/false positives/negatives for each class
            # For the first class
            if case[2] == 1: #acq 
                if estimate > 0:
                    print("Correct")
                    a_tp += 1
                    b_tn += 1
                else:
                    print("Wrong")
                    a_fn += 1
                    b_fp += 1
            # For the second class
            else:
                if estimate < 0:
                    print("Correct")
                    b_tp += 1
                    a_tn += 1
                else:
                    print("Wrong")
                    b_fn += 1
                    a_fp += 1

        precision_a = self.precision_a = a_tp/(a_tp+a_fp)
        recall_a = self.recall_a = a_tp/(a_tp+a_fn)
        f1_a = self.f1_a = 2*((precision_a*recall_a)/(precision_a+recall_a))
        precision_b = self.precision_b = b_tp/(b_tp+b_fp)
        recall_b = self.recall_b = b_tp/(b_tp+b_fn)
        f1_b = self.f1_b = 2*((precision_b*recall_b)/(precision_b+recall_b))
        if verbose:
            print("precision a " + str(precision_a))
            print("recall a " + str(recall_a))
            print("f1 a " + str(f1_a))
            print("precision b " + str(precision_b))
            print("recall b " + str(recall_b))
            print("f1 b " + str(f1_b))

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

    def print_results(self):
        '''Print results for this Kernel'''
        '''
        print("F1 score: ", self.f1_score)
        print("Precision: ", self.precision)
        print("Recall: ", self.recall)
        '''


    def __repr__(self):
        return "i'm a SSK!"

if __name__ == '__main__':
    cat_a = input("Name of category A: ")
    cat_b = input("Name of category B: ")
    cat_a_tr_c = int(input("Number of training samples from category A: "))
    cat_b_tr_c = int(input("Number of training samples from category B: "))
    cat_a_tst_c = int(input("Number of testing samples from category A: "))
    cat_b_tst_c = int(input("Number of testing samples from category B: "))
    lamda = float(input("Lambda value: "))
    if input('Running a specific case (1) or a test run? (2): ') == "1":
        max_features = int(input("Number of features: "))
        feature_length = int(input("length of features: "))
        ssk = SSK(cat_a, cat_b, max_features, feature_length, lamda, cat_a_tr_c,
            cat_a_tst_c, cat_b_tr_c, cat_b_tst_c, run_test=False)
        ssk.set_matrix()
        ssk.predict()
        ssk.print_kernel()
    else:
        max_features = int(input("Number of features: "))
        feature_length = int(input("Initial length of features: "))
        it_count = int(input("number of iterations: "))
        for i in range(it_count):
            print("run for length of feature: ", feature_length)
            time_init = time.time()
            ssk = SSK(cat_a, cat_b, max_features, feature_length, lamda, cat_a_tr_c,
                cat_a_tst_c, cat_b_tr_c, cat_b_tst_c, run_test=False)
            ssk.set_matrix()
            print("Feature fetching (sec): ", time.time()-time_init)
            ssk.predict()
            print("Prediction (sec): ", time.time()-time_init)
            ssk.print_kernel()
            ssk.print_results()
            feature_length+=1