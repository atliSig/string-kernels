import nltk

nltk
sentence = "We are learning at KTH and sometimes it is a lot of fun"

#tokenize
tokens = nltk.word_tokenize(sentence)

print(tokens)