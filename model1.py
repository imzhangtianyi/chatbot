import joblib
from flask import Flask
from flask import render_template,jsonify,request
import requests
from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

d = {'The Lion King':{0:"Sorry, We can't select seats as we are 3rd party vendors. But our system will automatically select the best seats available and we always book the seats together.",19:'There is no refund policy found'}, 'The Phantom of the Opera':{0:'Our intelligent system chooses the best seats available for the price you have paid in the section you have requested.',19:'## Cancellation Policy \r\nAll Broadway and Off-Broadway show tickets are non-refundable.'}}
# Tokenize chats
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')

# stop words
stop_words = stopwords.words('english')
stop_words.extend(['help', 'headout', 'hi', 'hello', 'hey', 'welcome', 'thank', 'check', 'please', 'ok','get', 'yes', 'safari', 'khalifa', 'burj', 'aquarium', 'roman', 'palatin', 'vatican', 'dubai'])
def remove_stopwords(l):
    return [word for word in l if word not in stop_words]

# stemming and lemmatizing
porter_stemmer = PorterStemmer()
def stem_words(l):
    return [porter_stemmer.stem(words) for words in l]

wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize_words(l):
    return [wordnet_lemmatizer.lemmatize(words) for words in l]

def process_text(text):
    return lemmatize_words(stem_words(remove_stopwords(tokenizer.tokenize(text.lower()))))


# import product model
rf = joblib.load('./model/Product_Model.sav')
le_p = joblib.load('./model/Product_Encoder.sav')
tv_p = joblib.load('./model/Product_Vectorizer.sav')

# import query model
sgd = joblib.load('./model/Query_Model.sav')
le_q = joblib.load('./model/Query_Encoder.sav')
tv_q = joblib.load('./model/Query_Vectorizer.sav')

# create a Flask app instance
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('home.html')

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
        if user_message == 'BADBOT':
            return jsonify({"status":"success","response":"I'm sorry. Let me redirect you to a real person..."})
#         Predict query
        answer_no = sgd.predict(tv_q.transform([user_message]))[0]
        product_name = cache.get("foo")
        if (answer_no not in [0, 19]):
            return jsonify({"status":"success","response":"I'm sorry, I have not trained for this question yet."})
        else:
            cache.set("answer", answer_no)
            
        answer_no = cache.get("answer")
#         Predict product
        product_t = tv_p.transform([user_message])
        product_no = rf.predict(product_t)
        if rf.predict_proba(product_t)[0][product_no][0] > 0.3:
            product_name = le_p.inverse_transform([product_no])[0][0]
            cache.set("foo", product_name)
        
#         product_name = cache.get("foo")

        if product_name:
                response_text = "For {},\n{}".format(product_name, d[product_name][answer_no])
        else:
            response_text = "What product are you looking at?"
        return jsonify({"status":"success","response":response_text})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Code error"})

if __name__ == '__main__':
    app.run(debug=True)