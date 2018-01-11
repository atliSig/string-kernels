import nltk
import os
from nltk.corpus import treebank

sentence = "We are learning at KTH and sometimes it is a lot of fun"

#tokenize
tokens = nltk.word_tokenize(sentence)
print(tokens)

#display a parse tree
t = treebank.parsed_sents('wsj_0005.mrg')[0]
t.draw()