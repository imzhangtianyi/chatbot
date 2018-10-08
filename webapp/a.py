import joblib
import json
from flask import Flask
from flask import render_template,jsonify,request
from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

with open('./model/product_db.json', 'r') as fp:
    product_db = json.load(fp)


# text preprocessing
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


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

    
# import product model
rf = joblib.load('./model/Product_Model.sav')
le_p = joblib.load('./model/Product_Encoder.sav')
tv_p = joblib.load('./model/Product_Vectorizer.sav')

# import query model
nb = joblib.load('./model/Topic_Model.sav')
tv_q = joblib.load('./model/Topic_Vectorizer.sav')

# create a Flask app instance
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
        user_message = preprocess(user_message).transform()

        if user_message == 'badbot':
            return jsonify({"status":"success","response":"I'm sorry :( Let me direct you to our agent."})

        if user_message in ['hi', 'hey', 'hello']:
            return jsonify({"status":"success","response":"Welcome to Headout! I'm Re:BOT, your customer service chatbot. If I failed to answer your question, please type 'BADBOT' to redirect to our agent."})


#         Predict query
        answer_no = nb.predict(tv_q.transform([user_message]))[0]
        answer_proba = nb.predict_proba(tv_q.transform([user_message])).max()

        if answer_proba < 0.9: # The minimum proba to understan a topic is 0.9
            answer_no = cache.get("answer")
        else:
            cache.set("answer", answer_no)

        product_name = cache.get("product")
        last_answer = cache.get("last")

        # if (answer_no not in [ 2,  5,  6,  9, 10, 11, 14, 15, 18, 19]):
        #     return jsonify({"status":"success","response":"Let me direct you to our local product specialist..."})
        # else:
        #     cache.set("answer", answer_no)

#         Predict product
        product_t = tv_p.transform([user_message])
        product_no = rf.predict(product_t)
        if rf.predict_proba(product_t)[0][product_no][0] > 0.5:
            product_name = le_p.inverse_transform([product_no])[0][0]
            cache.set("product", product_name)
        

        if product_name:
                response_text = "For {},\n{}".format(product_name, product_db[product_name][str(int(answer_no))])
                cache.set("last", response_text)
        else:
            response_text = "What product are you looking at?"

        if not answer_no:
            return jsonify({"status": "success",
                            "response": "Sorry, I didn't understand. Could you rephrase your question please?"})

        if response_text != last_answer:
            return jsonify({"status":"success","response": response_text})
        else:
            return jsonify({"status": "success",
                            "response": "Sorry, I'm not able to answer this. I can direct you to our local product specialist if you wish"})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response": "Code error"})

