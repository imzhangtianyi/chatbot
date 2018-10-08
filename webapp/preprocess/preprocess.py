from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class preprocess(object):
    '''
    Preprocessing text for classifier

    Parameters
    ----------
    text: text to be processed
    stop_words: list of stop_words from nltk
    '''

    def __init__(self, text):
        self.text = text

    def tokenizer(self):
        return RegexpTokenizer(r'\w+').tokenize(self.text.lower())

    def stem_words(self):
        '''
        Stem words

        Input:
        -----------------------
        List of words

        Return:
        ------------------------
        List of stemmed words
        '''
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(words) for words in self.tokenizer()]

    def lemmatize_words(self):
        '''
        Lemmatize words

        Input:
        -----------------------
        List of words

        Return:
        ------------------------
        List of lemmatized words
        '''
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer.lemmatize(words) for words in self.stem_words()]

    def transform(self):
        '''
        Transform and combine
        '''
        return ' '.join(self.lemmatize_words())
