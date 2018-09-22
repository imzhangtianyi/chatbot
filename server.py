from flask import Flask, request

# Import request
import requests

# create a Flask app instance
app = Flask(__name__)

# Tokens from the Facebook page web hooks
ACCESS_TOKEN = "EAAEhntNrh4kBAJuZAZCjd0Bo4bI4ubkfwNAyOptY1vtoZBgfruRwj4SAsEZCZArdoYOBW3EBvRWpDuZC1lZC2CxnYsO1IwWVm8TpyLUZC6UI94OcpVwZBA7yQs9fhBPDHZBBMiYW908X0phgDuKkmLYOvQfqSzwHTwjDxen3Lj3aA1Eo1HdDi43JhYYB63S32zhecZD"
VERIFY_TOKEN = "1234"

# method to reply to a message from the sender
def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    # Post request using the Facebook Graph API v2.6
    resp = requests.post("https://graph.facebook.com/v2.6/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print(resp.content)

# GET request to handle the verification of tokens
@app.route('/', methods=['GET'])
def handle_verification():
    if request.args['hub.verify_token'] == VERIFY_TOKEN:
        return request.args['hub.challenge']
    else:
        return "Invalid verification token"

# POST request to handle in coming messages then call reply()
@app.route('/', methods=['POST'])
def handle_incoming_messages():
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    message = data['entry'][0]['messaging'][0]['message']['text']
    reply(sender, message)

    return "ok"

# Run the application.
if __name__ == '__main__':
    app.run(debug=True)