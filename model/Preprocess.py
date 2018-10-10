"""
Stem, lemmatize, and Remove Stopwords
"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



def tokenizer(text):
    return RegexpTokenizer(r'\w+').tokenize(text.lower())


def stem_words(tokens):
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
    return [porter_stemmer.stem(words) for words in tokens]


def lemmatize_words(tokens):
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
    return [wordnet_lemmatizer.lemmatize(words) for words in tokens]


def prep(text, stop_words):
    '''
    Remove stop words

    Input:
    -----------------------
    List of words

    Return:
    ------------------------
    List of words with stop words removed
    '''
    tokens = tokenizer(text)
    stem_list = stem_words(tokens)
    lemma_list = lemmatize_words(stem_list)
    return [word for word in lemma_list if word not in stop_words]
