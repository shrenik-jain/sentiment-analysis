import numpy as np 
from collections import Counter
from utils.dataclean import DataClean
from utils.vocabulary import Vocabulary
from utils.model import Model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

vocab = Counter()
v = Vocabulary()
c = DataClean() 
m = Model()

#Creating a Vocabulary
tokens = v.preprocess('reviews/train/negative', vocab, True)
tokens = v.preprocess('reviews/train/positive', vocab, True)
v.saver(tokens, 'vocab.txt')

filename = 'vocab.txt'
vocab = v.load_doc(filename)
vocab = vocab.split()
vocab = set(vocab)

##########################################################################################

#Cleaning the train reviews based on the vocabulary
train_positive_docs = c.process_docs('reviews/train/positive', vocab, True)
train_negative_docs = c.process_docs('reviews/train/negative', vocab, True)
train_docs = train_negative_docs + train_positive_docs

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

train_encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = max([len(s.split()) for s in train_docs])

X_train = pad_sequences(train_encoded_docs, maxlen = max_length, padding = 'post')
y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

##########################################################################################

# Cleaning the test reviews based on the vocabulary
test_positive_docs = c.process_docs('reviews/test/positive', vocab, False)
test_negative_docs = c.process_docs('reviews/test/negative', vocab, False)
test_docs = test_negative_docs + test_positive_docs

test_encoded_docs = tokenizer.texts_to_sequences(test_docs)

X_test = pad_sequences(test_encoded_docs, maxlen = max_length, padding = 'post')
y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

##########################################################################################

vocab_size = len(tokenizer.word_index) + 1
m.train(X_train, y_train, X_test, y_test, vocab_size, max_length)
