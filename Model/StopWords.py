"""
Stop words used in the topic models
"""
import pandas as pd
from nltk import corpus

stop_words = corpus.stopwords.words('english') # stop words

def common():
    """
    Only contains common words
    :return:
    stop words with common words
    """
    stop_words.extend(
        ['hello', 'hi', 'welcom', 'headout', 'know', 'experi', 'refer', 'help', 'ani', 'chat', 'problem', 'may',
         'reach', 'need', 'let', 'u', 'feel', 'free', 'contact', 'realli', 'appreci', 'could', 'rate', 'chat', 'thank',
         'today', 'wa', 'nice', 'talk', 'great', 'day', 'goodby', 'would', 'like', 'plea', 'wait', 'minut', 'check',
         'thi', 'anyth', 'el', 'step', 'away', 'assist', 'custom', 'bye', 'hey', 'ok', 'get', 'ye', 'safari', 'khalifa',
         'burj', 'aquarium', 'roman', 'palatin', 'vatican', 'dubai']
    )
    return stop_words

def products():
    """
    Contains popular product names
    :return:
    stop words with product names
    """
    stop_list = pd.read_csv('stop_words_product.csv')[4:120] # Load the high feature importance words from product model
    stop_words.extend(stop_list.feature.values.tolist())
    return stop_words

def anchor():
    # Anchor Words for Anchored CorEx
    anchor_words = [['cashback', 'cash'],
                    ['refund', 'cancel'],
                    ['child', 'adult', 'year', 'old', 'age', 'kid'],
                    ['seat', 'choos', 'select', 'section', 'offic', 'exact', 'togeth'],
                    ['discount', 'coupon', 'code', 'offer'],
                    ['card', 'payment', 'work', 'complet', 'error', 'issu', 'differ', 'tri'],
                    ['pm', 'morn', 'night']
                    ]
    return anchor_words