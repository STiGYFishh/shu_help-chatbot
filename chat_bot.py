import chat_bot_predict as predict
import json
from flask import Flask, abort, request

app = Flask(__name__)

app.secret_key = 'ABC'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/api/v1/query', methods=['GET'])
def query():
    if not request.json:
        abort(400)
    data = json.loads(request.data)
    response, context = predict.response(data['query'], data['context'])
    return {"response": response, "context": context}
