"""
Apply the Anchored CorEx topic model on text
For Insight project Re:BOT
@Author: Tianyi Zhang
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Import CorEx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct

# Import text preprocessing functions
import Preprocess
import StopWords

df_t = pd.read_csv('QandA.csv')
stop_words = StopWords.common()
# Preprocess transcripts
df_l = df_t[:10000].QandA.apply(lambda s: ' '.join(Preprocess.prep(s, stop_words)))

# Anchor words
anchor_words = StopWords.anchor()

class Topics(object):
    """
    Topic modeling using Anchored CorEx
    """
    def __init__(self, text, anchors=anchor_words, n_topic=25, max_features=20000, max_iter=200, seed=100, anchor_strength=10):
        """
        Initialize and train the CorEx model
        :param text: A text series of customer service transcripts
        :param n_topic: number of topics
        :param max_features: maximum features for word embedding
        :param max_iter: maximum iteration time
        :param seed: random state
        :param anchor_strength: weight for anchor word
        """
        self.text = text
        self.n_topic = n_topic
        cv = CountVectorizer(stop_words='english', max_features=max_features, binary=True)
        # Corpus
        corpus = cv.fit_transform(text)
        # Vocabulary
        words = list(np.asarray(cv.get_feature_names()))
        # Build model
        self.topic_model = ct.Corex(n_hidden=n_topic, max_iter=max_iter, seed=seed)
        self.topic_model.fit(corpus, words=words, anchors=anchors, anchor_strength=anchor_strength)
        self.topics = self.topic_model.get_topics()

    def print_topics(self, max_topics=25):
        """
        Print topics
        :param max_topics: maximum number of topics to show
        :return: topic words
        """
        for n, topic in enumerate(self.topics):
            if n <= max_topics:
                topic_words, _ = zip(*topic)
                print('{}: '.format(n) + ','.join(topic_words))
            else:
                break

    def doc_topic(self):
        """
        Topics for the entire document
        :return:
        A pandas dataframe of topics with 3 columns
        logsum: sum of total topic correlation for each topic
        PTC: Pointwise total topic correlation
        """
        # Documents topic dataframe
        df_top = pd.DataFrame(columns=['logsum', 'topic', 'PTC'])
        TC = self.topic_model.log_z # total topic correlation
        df_top['logsum'] = TC.sum(1) # sum of total topic correlation
        df_top['topic'] = TC.argmax(1)
        # Calculate Pointwise TC for each document
        doc_cor = []
        for i, row in df_top.iterrows():
            doc_cor.append(TC[i, int(row.topic)])
        df_top['PTC'] = doc_cor
        return df_top

    def topic_percentage(self, PTC=8):
        """
        Percentage for each topics
        :param PTC: PTC threshold
        :return: A pandas dataframe with 3 columns, topic number, percentage, and topic key words
        """
        # Percentage of each topic
        df_top = self.doc_topic()
        df_perc = pd.DataFrame(columns=['topic_no', 'percentage', 'keywords'])
        df_perc['topic_no'] = list(i for i in range(self.n_topic))
        rowN = float(df_top.shape[0])
        for i in range(self.n_topic):
            #     Calculate percentage
            df_perc.at[i, 'percentage'] = df_top.loc[(df_top.topic == i) & (df_top.PTC > 8)].shape[0] / rowN
            #     Key words
            topic_words, _ = zip(*self.topics[i])
            df_perc.at[i, 'keywords'] = ' ,'.join(topic_words)

        return df_perc

    def roc(self):
        """
        train a classifier and plot ROC curves
        :return: trained model
        """
        df_top = self.doc_topic()
        stop_words = StopWords.products()
        # Prepare train set: df_q ===> x; Class label: y
        df_q = self.text.loc[(df_top.topic < 7) & (df_top.PTC > 8)]
        df_q = df_q.apply(lambda s: ' '.join(Preprocess.prep(s, stop_words)))
        y = df_top.loc[(df_top.topic < 7) & (df_top.PTC > 8)].topic

        # Tfidf
        tv = TfidfVectorizer(ngram_range=(1, 2), max_df=4000, min_df=10)
        x = tv.fit_transform(df_q)

        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score

        # Cross validation
        nb = MultinomialNB(alpha=0.05)
        yp = cross_val_predict(nb, x, y, cv=5, method='predict_proba')

        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        for i in range(7):
            fpr, tpr, _ = roc_curve(y == i, yp[:, i])
            auc = roc_auc_score(y == i, yp[:, i])
            plt.plot(fpr, tpr, label='Topic%d: AUC=%.2f' % (i, auc))
            plt.legend(loc='lower right', fontsize=12)
        plt.plot([0, 1], [0, 1], '--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('1 - specificity', fontsize=14)
        plt.ylabel('sensitivity', fontsize=14)
        plt.show()

        nb = MultinomialNB(alpha=0.05)
        nb.fit(x, y)
        return nb



if __name__ == "__main__":
    Topics(df_l).roc()