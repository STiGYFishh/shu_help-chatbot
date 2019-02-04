import chat_bot_predict as predict
import json
from flask import Flask, session, abort, request

app = Flask(__name__)

app.secret_key = 'ABC'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/api/v1/query', methods=['POST'])
def query():
    if not request.json:
        abort(400)
    data = json.loads(request.data)
    response = predict.response(data['query'], session)
    return response
