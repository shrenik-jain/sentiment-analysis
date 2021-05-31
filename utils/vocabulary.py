from string import punctuation
from os import listdir

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


class Vocabulary:

    @staticmethod
    def loader(filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()

        return text


    @staticmethod
    def saver(lines, filename):
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()
    
    
    @staticmethod
    def cleaner(doc):

        tokens = doc.split()                            	   
        table = str.maketrans('', '', punctuation)              # removing punctuations
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]    # removing non-alphabetic tokens

        stop_words = set(stopwords.words('english'))            #removing stop words
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if len(word) > 1]     #removing single alphabet tokens

        return tokens
    

    def generate_vocab(self, filename, vocab):
        doc = self.loader(filename)
        tokens = self.cleaner(doc)
        vocab.update(tokens)

    
    @staticmethod
    def update_vocab(vocab, min_occurance):
        tokens = [k for k,c in vocab.items() if c >= min_occurance]  #remove rarely occuring words
        
        return tokens
    

    def preprocess(self, directory, vocab, is_trian):

        for filename in listdir(directory):
            if is_trian and filename.startswith('cv9'):
                continue
            if not is_trian and not filename.startswith('cv9'):
                continue

            path = directory + '/' + filename
            self.generate_vocab(path, vocab)

        tokens = self.update_vocab(vocab, 2)

        return tokens