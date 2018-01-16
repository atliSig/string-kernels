'''An SSK lazy implementation'''
import re
import random
import sys
import time
import numpy as np
from nltk.corpus import reuters, stopwords
from collections import defaultdict
from operator import getitem
from cvxopt.solvers import qp
import cvxopt.solvers as cvx_solver
from cvxopt import matrix
cvx_solver.options['show_progress'] = False
from itertools import combinations

class Document:
    '''A class for a document from the Reuters data-set'''
    def __init__(self, category, index, m, n, contigous=True, blob_length=200):
        '''
            :param category: the name of the document's category
            :param m: the number of top features
            :param n: the length of each feature
            :param index: the index of the document into the Reuters data-set
            :param contigous: Boolean, True if features are contigous, True otherwise 
        '''

        self.m = m
        self.n = n
        self.index = index
        self.category = category
        self.contigous = contigous
        if self.contigous:
            self.features = set()
            self.words = reuters.words(index)
        else:
            self.features = set()
            self.blob_length = blob_length
            self.noncont_features = defaultdict(lambda: {'count':0, 'weights':[]}, {})
            self.words = reuters.words(index)[:self.blob_length]
        self.clean_data = self.remove_stops()
        self.set_features()
        self.sort_features()
    
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
        if self.contigous:
            for i in range(len(self.clean_data)-self.n+1):
                self.features.add(self.clean_data[i:i+self.n])
        else:
            # get all features, including non-congtigous that appear
            # within the same word
            for word in self.clean_data.split(' '):
                if len(word) > self.n:
                    comb = combinations(range(len(word)), self.n)
                    for c in comb:
                        w = ''.join([word[i] for i in c])
                        self.features.add(w)
                        self.noncont_features[w]['count']+=1
                        self.noncont_features[w]['weights'].append(c[-1]-c[0]+1)

    def sort_features(self):
        '''returns features in the order of number of occurrences'''
        tuples = {}
        for f in self.features:
            tuples[f] = self.clean_data.count(f)
        tuples_sorted = sorted(tuples, key=tuples.get, reverse=True)
        # self.freq_features is of type list
        self.freq_features = tuples_sorted
        if not self.contigous:
            self.noncont_freq_features = sorted(self.noncont_features.items(), key=lambda x:getitem(x[1],'count'),
                reverse=True)

    def get_top_features(self):
        '''Returns the list of top features for this Document'''
        return self.freq_features[:self.m]
    
    def __repr__(self):
        return 'Doc: '+ self.index +' in category: ' + self.category

class SSK:
    '''A class for a lazy SSK implementation'''
    def __init__(self, cat_a, cat_b, max_features, k, lamda, cat_a_tr_c=0, cat_a_tst_c=0,
        cat_b_tr_c=0, cat_b_tst_c=0, avg_it=5, threshold=10**-5, seed=None, contigous=True, ngram=False):
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
            :param contigous: Boolean, True if features are contigous, True otherwise 
        '''
        self.k = k
        self.avg_it = avg_it
        self.lamda = lamda
        self.max_features = max_features
        self.threshold = threshold
        self.cat_a = cat_a
        self.cat_b = cat_b
        self.contigous = contigous
        self.ngram = ngram
        self.label = {
            self.cat_a:1,
            self.cat_b:-1
        }
        # a list of document indeces for both categories
        self.docs_a = reuters.fileids(cat_a)
        self.docs_b = reuters.fileids(cat_b)
        # the length of those categories
        self.cat_a_count = len(self.docs_a)
        self.cat_b_count = len(self.docs_b)
        # the complete lists of training/testing documents in both categories
        self.cat_a_training = list(filter(lambda doc: doc.startswith("train"), self.docs_a))
        self.cat_a_testing = list(filter(lambda doc: doc.startswith("test"), self.docs_a))
        self.cat_b_training = list(filter(lambda doc: doc.startswith("train"), self.docs_b))
        self.cat_b_testing = list(filter(lambda doc: doc.startswith("test"), self.docs_b))
        # the number of training/testing samples for this SSK
        self.cat_a_tr_c = min(cat_a_tr_c, len(self.cat_a_training))
        self.cat_a_tst_c = min(cat_a_tst_c, len(self.cat_a_testing))
        self.cat_b_tr_c = min(cat_b_tr_c, len(self.cat_b_training))
        self.cat_b_tst_c = min(cat_b_tst_c, len(self.cat_b_testing))
        # A list of objects of type Document for both training/testing
        # including both categories
        self.training_list = []
        self.testing_list = []

        if cat_a_tr_c+cat_a_tst_c > self.cat_a_count or cat_b_tr_c+cat_b_tst_c > self.cat_b_count:
            print('number of training/testing documents exceeds number of articles')
            sys.exit(0)

        for i in self.cat_a_training[:cat_a_tr_c]:
            self.training_list.append(Document(self.cat_a, i, self.max_features, self.k, contigous=self.contigous))

        for i in self.cat_b_training[:cat_b_tr_c]:
            self.training_list.append(Document(self.cat_b, i, self.max_features, self.k, contigous=self.contigous))

        for i in self.cat_a_testing[:cat_a_tst_c]:
            self.testing_list.append(Document(self.cat_a, i, self.max_features, self.k, contigous=self.contigous))

        for i in self.cat_b_testing[:cat_b_tst_c]:
            self.testing_list.append(Document(self.cat_b, i, self.max_features, self.k, contigous=self.contigous))

        self.kernel_matrix = np.zeros([cat_a_tr_c+cat_b_tr_c, cat_a_tr_c+cat_b_tr_c])
        self.top_feature_list = set()
        self.all_feature_list = set()
        self.seed = seed
        self.alpha_list = []

    def shuffle_train_test_data(self):
        m = len(self.training_list)
        all_data = self.training_list + self.testing_list
        random.shuffle(all_data)
        self.training_list, self.testing_list = all_data[:m], all_data[m:]


    def set_matrix(self):
        '''Create the matrix here'''
        # create list of lists where each inner-list is [1/-1,index]
        random.shuffle(self.training_list)
        random.shuffle(self.testing_list)

        for doc in self.training_list:
            self.top_feature_list.update(doc.get_top_features())
            self.all_feature_list.update(doc.features)

        for i in range(len(self.training_list)):
            for j in range(i,len(self.training_list)):
                sample_i = self.training_list[i]
                sample_j = self.training_list[j]
                if self.ngram:
                    self.kernel_matrix[i, j] = self.calc_kernel_ngram(sample_i, sample_j)*\
                    self.label[sample_i.category]*self.label[sample_j.category]
                else:
                    self.kernel_matrix[i, j] = self.calc_kernel(sample_i, sample_j)*\
                    self.label[sample_i.category]*self.label[sample_j.category]
                self.kernel_matrix[j, i] = self.kernel_matrix[i, j]

        #Normalizing results in rank issues with cvxopt.qp
        #self.normalize_kernel()

    def normalize_kernel(self):
        '''Frobenius-Normalization of the kernel'''
        for i in range(len(self.training_list)):
            for j in range(len(self.training_list)):
                self.kernel_matrix[i, j] = self.kernel_matrix[i, j]/\
                np.sqrt(self.kernel_matrix[i, i]*self.kernel_matrix[j, j])

    def predict(self):
        '''Based on the kernel, make predictions using cvxopt.qp'''
        G = -np.eye(len(self.training_list))
        G = np.append(G, np.eye(len(self.training_list)))
        G.resize(2 * len(self.training_list), len(self.training_list))
        # C is the slack
        C = 10
        h = np.zeros(len(self.training_list))
        h_alpha = np.ones(len(self.training_list)) * C

        h = np.append(h, h_alpha)
        h.resize(2 * len(self.training_list))
        q = -np.ones((len(self.training_list)))

        # Optimizes the alpha values, alpha is a 1-d list
        alpha = list(qp(matrix(self.kernel_matrix), matrix(q), matrix(G), matrix(h))['x'])
        # calculates the alphas that are larger than the threshold
        self.set_alpha(alpha, self.training_list)

    def set_alpha(self, alpha, training_docs):
        '''
            Sets the list of support vectors
            Returns a list of tuples where each tuple contains
            a document and the corresponding alpha value
        '''
        self.alpha_list = [[training_docs[idx], al_el] for idx,al_el in enumerate(alpha) if al_el > self.threshold]

    def get_alpha(self):
        '''Gets the list of support vectors'''
        return self.alpha_list
    
    def calc_kernel(self, doc_a, doc_b):
        '''Calculates the kernel matrix value for K[i,j]'''
        total = 0
        for feature in self.top_feature_list:
            if self.contigous:
                l = doc_a.clean_data.count(feature)
                j = doc_b.clean_data.count(feature)
                total += l*j*self.lamda**(2*self.k)
            else:
                weights_a = doc_a.noncont_features[feature]['weights']
                val_a = sum([self.lamda**w for w in weights_a])
                weights_b = doc_b.noncont_features[feature]['weights']
                val_b = sum([self.lamda**w for w in weights_b])
                total += val_a*val_b
        return total

    def calc_kernel_ngram(self, doc_1, doc_2):
        '''Calculates the kernel value for the ngram version'''
        shared_ngrams = set()
        shared_ngrams.update(doc_1.features)
        shared_ngrams.update(doc_2.features)
        total = 0
        for n_gram in shared_ngrams:
            total += doc_1.clean_data.count(n_gram) * doc_2.clean_data.count(n_gram)
        return total

    def ind(self, doc):
        '''
            takes in a document and calculates
            a * b * calc_kernel(c, doc)
            where:
                a = alpha value
                b = label of the document
                c = the document of a support vector
        '''
        if(self.ngram):
            return np.sum([a[1] * self.label[a[0].category] *\
                self.calc_kernel_ngram(a[0], doc) for a in self.alpha_list])
        else:
            return np.sum([a[1] * self.label[a[0].category] * \
                self.calc_kernel(a[0], doc) for a in self.alpha_list])
        
    def set_results(self, verbose=True):
        '''Print results for this Kernel'''
        # Class a is assigned positive values and B negative values
        a_tp = a_tn = a_fp = a_fn = b_tp = b_tn = b_fp = b_fn = 0
        for doc in self.testing_list:
            estimate = self.ind(doc)
            #check for true/false positives/negatives for each class
            if doc.category == self.cat_a:
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
        return 

if __name__ == '__main__':
    cat_a = input("Name of category A (default earn): ") or "earn"
    cat_b = input("Name of category B (default acq): ") or "acq"
    cat_a_tr_c = int(input("Number of training samples from category A (default 152): ") or 152)
    cat_b_tr_c = int(input("Number of training samples from category B (default 114): ") or 114)
    cat_a_tst_c = int(input("Number of testing samples from category A (default 40): ") or 40)
    cat_b_tst_c = int(input("Number of testing samples from category B (default 25): ") or 25)
    lamda = float(input("Lambda value (default 1.0): ") or 1)
    threshold = float(input("Threshold value (default 0.00001):") or 10**-5)
    max_features = int(input("Number of features (default 30): ") or 30)
    feature_it = input("number of different length of features (default [3,..,8,10,12,14]): ")\
        or [3, 4, 5, 6]
    avg_it = int(input("number of iterations (default 10): ") or 10)
    non_contigous = input("Are strings non-contigous ([True,False], default: False)?: ")
    non_contigous = (non_contigous == "True")
    ngram = input("Use ngram version? ([True,False], default: False)?: ")
    ngram = (ngram == "True")
    verbose_time = input("Print updates ([True,False], default: False): ")
    verbose_time = (verbose_time == "True")
    output_labels = ['precision_a', 'f1_a', 'recall_a', 'precision_b', 'f1_b', 'recall_b']
    if lamda <= 0 or lamda > 1:
        print('lamda must be in ]0,1]')
        sys.exit(0)

    result_matrix = np.zeros((len(feature_it), len(output_labels)*2))
    for idx_feat, feat, in enumerate(feature_it):
        outputs = []
        outer_loop_time = time.time()
        for j in range(avg_it):
            time_init = time.time()
            print("Starting creation of SSK")
            ssk = SSK(cat_a, cat_b, max_features, feat, lamda, cat_a_tr_c,
                      cat_a_tst_c, cat_b_tr_c, cat_b_tst_c, avg_it, threshold, contigous=(not non_contigous), ngram=ngram)
            ssk.set_matrix()
            ssk.shuffle_train_test_data()
            print("Done with ssk.set_matrix()")
            if verbose_time:
                print("run for length of feature: ", feat)
                time_secondary = time.time()
                print("Feature fetching (sec): ", time.time()-time_init)
            ssk.predict()
            ssk.set_results(verbose=False)
            if verbose_time:
                print("Prediction (sec): ", time.time()-time_secondary)
            print("Results for iteration: "+ str(j) +", for feature length: " +str(feat))
            print(ssk.get_results(verbose=False))
            outputs.append(ssk.get_results(verbose=False))

        avg_list = [np.average([c[i] for c in outputs]) for i in range(len(output_labels))]
        std_list = [np.std([c[i] for c in outputs]) for i in range(len(output_labels))]
        out = []
        for i in range(len(avg_list)):
            out.append(avg_list[i])
            out.append(std_list[i])
        result_matrix[idx_feat,:] = out
    # write results
    print(result_matrix)
    print(len(feature_it))
    with open('out.txt','wb') as f:
        np.savetxt(f, result_matrix, fmt='%.5f')
