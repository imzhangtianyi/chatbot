# Ref: https://github.com/bhavaniravi/rasa-site-bot
from flask import Flask
from flask import render_template,jsonify,request
import requests
import random


app = Flask(__name__)
app.secret_key = '12345'

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/chat',methods=["POST"])
def chat():
    user_message = request.form["text"]
    return jsonify({"status":"success","response":user_message})


if __name__ == "__main__":
    app.run(port=8080)
