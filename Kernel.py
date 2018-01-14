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
        print(index)
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
        self.docs_a = reuters.fileids(cat_a)
        self.docs_b = reuters.fileids(cat_b)
        self.cat_a_count = len(self.docs_a)
        self.cat_b_count = len(self.docs_b)
        self.cat_a_tr_c = cat_a_tr_c
        self.cat_a_tst_c = cat_a_tst_c
        self.cat_b_tr_c = cat_b_tr_c
        self.cat_b_tst_c = cat_b_tst_c
        self.run_test = run_test

        if self.run_test:
            self.training_list = [[self.cat_a, i, -1] for i in
                                  range(self.cat_a_tr_c)] + [[self.cat_b, i, 1] for i in range(self.cat_b_tr_c)]

            self.testing_list = [[self.cat_a, i, -1]
                                 for i in range(self.cat_a_tr_c + 1, self.cat_a_tr_c + self.cat_a_tst_c + 1)] + \
                [[self.cat_b, i, 1] for i in
                 range(self.cat_b_tr_c + 1, self.cat_b_tr_c + self.cat_b_tst_c + 1)]
            #training_list_a = list(filter(lambda doc: doc.startswith("train"), self.docs_a))[:self.cat_a_tr_c]
            #training_list_b = list(filter(lambda doc: doc.startswith("train"), self.docs_b))[:self.cat_a_tr_c]
            #testing_list_a = list(filter(lambda doc: doc.startswith("test"), self.docs_a))[:self.cat_b_tst_c]
            #testing_list_b = list(filter(lambda doc: doc.startswith("test"), self.docs_b))[:self.cat_b_tst_c]
            #self.training_list = training_list_a + training_list_b
            #self.testing_list = testing_list_a + testing_list_b
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
            print(doc)
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
        # Class a is assigned positive values and B negative values
        a_tp = a_tn = a_fp = a_fn = b_tp = b_tn = b_fp = b_fn = 0
        for case in self.testing_list:
            estimate = self.ind(case, self.alpha_list_global)
            #check for true/false positives/negatives for each class
            # For the first class
            if case[2] == 1: #acq 
                if estimate > 0:
                    if verbose: print("Correct")
                    a_tp += 1
                    b_tn += 1
                else:
                    if verbose: print("Wrong")
                    a_fn += 1
                    b_fp += 1
            # For the second class
            else:
                if estimate < 0:
                    if verbose: print("Correct")
                    b_tp += 1
                    a_tn += 1
                else:
                    if verbose: print("Wrong")
                    b_fn += 1
                    a_fp += 1

        self.precision_a = a_tp/(a_tp+a_fp)
        self.recall_a = a_tp/(a_tp+a_fn)
        self.f1_a = 2*((self.precision_a*self.recall_a)/(self.precision_a+self.recall_a))
        self.precision_b = b_tp/(b_tp+b_fp)
        self.recall_b = b_tp/(b_tp+b_fn)
        self.f1_b = 2*((self.precision_b*self.recall_b)/(self.precision_b+self.recall_b))

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

    def get_results(self, verbose=True):
        if verbose:
            print("precision a " + str(self.precision_a))
            print("recall a " + str(self.recall_a))
            print("f1 a " + str(self.f1_a))
            print("precision b " + str(self.precision_b))
            print("recall b " + str(self.recall_b))
            print("f1 b " + str(self.f1_b))
        return [self.precision_a, self.f1_a, self.recall_a, 
            self.precision_b, self.f1_b, self.recall_b]


    def __repr__(self):
        return "i'm a SSK!"

if __name__ == '__main__':
    cat_a = input("Name of category A (default corn): ") or "corn"
    cat_b = input("Name of category B (default earn): ") or "earn"
    cat_a_tr_c = int(input("Number of training samples from category A (default 10): ") or 10)
    cat_b_tr_c = int(input("Number of training samples from category B (default 10): ") or 10)
    cat_a_tst_c = int(input("Number of testing samples from category A (default 10): ") or 10)
    cat_b_tst_c = int(input("Number of testing samples from category B (default 10): ") or 10)
    lamda = float(input("Lambda value (1.0): ") or 1)
    if input('Running a specific case (y/n)? (default n) ') == "y":
        max_features = int(input("Number of features (default 10): ") or 10)
        feature_length = int(input("length of features default (default 4): ") or 4)
        ssk = SSK(cat_a, cat_b, max_features, feature_length, lamda, cat_a_tr_c,
            cat_a_tst_c, cat_b_tr_c, cat_b_tst_c, run_test=False)
        ssk.set_matrix()
        ssk.predict()
        ssk.print_kernel()
    else:
        max_features = int(input("Number of features (default 10): ") or 10)
        feature_length = int(input("length of features default (default 4): ") or 4)
        feature_it = int(input("number of different length of features (default 5): ") or 5)
        avg_it = int(input("number of iterations (default 5): ") or 5)
        output_labels = ['precision_a', 'f1_a', 'recall_a', 'precision_b', 'f1_b', 'recall_b']
        for i in range(feature_it):
            outputs = []
            outer_loop_time = time.time()
            for j in range(avg_it):
                print("run for length of feature: ", feature_length)
                time_init = time.time()
                ssk = SSK(cat_a, cat_b, max_features, feature_length, lamda, cat_a_tr_c,
                    cat_a_tst_c, cat_b_tr_c, cat_b_tst_c, run_test=True)
                ssk.set_matrix()
                time_secondary = time.time()
                print("Feature fetching (sec): ", time.time()-time_init)
                ssk.predict()
                print("Prediction (sec): ", time.time()-time_secondary)
                #ssk.print_kernel()
                ssk.set_results()
                outputs.append(ssk.get_results(verbose=False))
                feature_length+=1
            print("Total time for the current feature: ", time.time()-outer_loop_time)
            print("Here come the results: ")
            for i in range(len(output_labels)):
                curr = [c[i] for c in outputs]
                print("average for"+output_labels[i],np.average(curr))
