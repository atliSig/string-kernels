import nltk
import os
from nltk.corpus import treebank
import itertools

sentence = "We are, us 4, learning at KTH. Sometimes it is a lot of fun."

#tokenize
tokens = nltk.word_tokenize(sentence)
print(tokens)

tokens = nltk.regexp_tokenize(sentence, r'[,\.]\s*', gaps=False)
print(tokens)

#display a parse tree
#t = treebank.parsed_sents('wsj_0005.mrg')[0]
#t.draw()


print(list(itertools.combinations(['cat','car'], 2)))