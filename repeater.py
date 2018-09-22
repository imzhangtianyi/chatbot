from flask import Flask
from flask import render_template,jsonify,request
import requests

# create a Flask app instance
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('home.html')

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
#         response = requests.get("http://localhost:5000/parse",params={"q":user_message})
#         response = response.json()
#         entities = response.get("entities")
#         topresponse = response["topScoringIntent"]
#         intent = topresponse.get("intent")
        response_text = user_message
        return jsonify({"status":"success","response":response_text})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Redirecting to a human..."})

if __name__ == '__main__':
    app.run(debug=True)