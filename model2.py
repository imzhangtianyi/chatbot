import joblib
import json
from flask import Flask
from flask import render_template,jsonify,request
import requests
from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

with open('./model/product_j.json', 'r') as fp:
    product_db = json.load(fp)



# import product model
rf = joblib.load('./model/Product_Model_nb.sav')
le_p = joblib.load('./model/Product_Encoder_nb.sav')
tv_p = joblib.load('./model/Product_Vectorizer_nb.sav')

# import query model
sgd = joblib.load('./model/Query_Model.sav')
tv_q = joblib.load('./model/Query_Vectorizer.sav')

# create a Flask app instance
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]

        if user_message == 'BADBOT':
            return jsonify({"status":"success","response":"I'm sorry. Let me redirect you to a real person..."})

        if user_message.lower() in ['hi', 'hey', 'hello']:
            return jsonify({"status":"success","response":"Welcome to Headout! I'm Frost, your customer service chatbot. If I failed to answer your question, please type 'BADBOT' to redirect to our agent."})


#         Predict query
        answer_no = sgd.predict(tv_q.transform([user_message]))[0]
        if sgd.decision_function(tv_q.transform([user_message])).max() < 0:
            answer_no = cache.get("answer")

        product_name = cache.get("product")
        if (answer_no not in [ 2,  5,  6,  9, 10, 11, 14, 15, 18, 19]):
            return jsonify({"status":"success","response":"Let me direct you to our local product specialist..."})
        else:
            cache.set("answer", answer_no)
            
        answer_no = cache.get("answer")
#         Predict product
        product_t = tv_p.transform([user_message])
        product_no = rf.predict(product_t)
        if rf.predict_proba(product_t)[0][product_no][0] > 0.5:
            product_name = le_p.inverse_transform([product_no])[0][0]
            cache.set("product", product_name)
        

        if product_name:
#                 response_text = str(answer_no)
                response_text = "For {},\n{}".format(product_name, product_db[product_name][str(int(answer_no))])
        else:
            response_text = "What product are you looking at?"
        return jsonify({"status":"success","response":response_text})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Code error"})

if __name__ == '__main__':
    app.run(debug=True)
