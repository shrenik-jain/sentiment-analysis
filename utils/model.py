from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class Model:

    @staticmethod
    def train(X_train, y_train, X_test, y_test, vocab_size, max_length):

        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length = max_length))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, verbose=2)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        print(f'\n\nTest Accuracy: {acc*100} %')