from string import punctuation
from os import listdir
from utils.vocabulary import Vocabulary
from nltk.corpus import stopwords

class DataClean(Vocabulary):

    @staticmethod
    def clean_doc(doc, vocab):
        tokens = doc.split()
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)

        return tokens


    def process_docs(self, directory, vocab, is_trian):
        documents = list()
        for filename in listdir(directory):
            if is_trian and filename.startswith('cv9'):
                continue
            if not is_trian and not filename.startswith('cv9'):
                continue
            path = directory + '/' + filename
            doc = super().loader(path)
            tokens = self.clean_doc(doc, vocab)
            documents.append(tokens)

        return documents
 