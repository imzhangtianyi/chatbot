from flask import Flask
from flask import render_template,jsonify,request
import requests
import random

app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/chat',methods=["POST"])
def chat():
    user_message = request.form["text"]
    if user_message == 'BADBOT':
        return jsonify({"status":"success","response":"I'm sorry. Let me redirect you to a real person..."})
    return jsonify({"status":"success","response":"Hi, I'm your chatbot, glad to be at your service.\n Did you just said, '{}'?".format(user_message)})


if __name__ == "__main__":
    app.run(port=8080)
