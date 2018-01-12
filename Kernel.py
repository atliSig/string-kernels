from nltk.corpus import reuters, stopwords
import operator

def main():
    # List of documents
    print(stopwords.words('english'))
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    print(str(len(train_docs)) + " total train documents");

    test_docs = list(filter(lambda doc: doc.startswith("test"),
                            documents));
    print(str(len(test_docs)) + " total test documents");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");

    # Documents in a category
    category_docs = reuters.fileids("earn");

    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);

    doc = removeStopwords(reuters.words(category_docs[0]))
    features = featureExtraction(doc,4)
    f = freqFeatures(features,doc)
    print(f);

    # Raw document
    print(reuters.raw(document_id));

def removeStopwords(document):
    specialCharacters = [".", ",", ":", ";", "_", "-", "&", "%", "<", ">", "!", "?", "="]
    cleaned = [x.lower() for x in document if x not in specialCharacters and x.lower() not in stopwords.words('english')]
    return ' '.join(cleaned)

def featureExtraction(document, k):
    features = set()
    for i in range(len(document)-k+1):
        features.add(document[i:i+k])
    return features

def freqFeatures(features, document):
    tuples = {}
    for f in features:
        tuples[f] = document.count(f)
    tuples_sorted = sorted(tuples, key=tuples.get, reverse=True)
    return tuples_sorted

main()


