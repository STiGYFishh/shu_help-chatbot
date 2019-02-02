import json
import os
import random
import tensorflow as tf
import tflearn
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Disable AVX CPU Support Warning, disable keep_dims deprecation warning.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

stemmer = LancasterStemmer()

with open('intents.json', 'r') as fh:
    intents = json.load(fh)

with open('training_data.json', 'r') as fh:
    data = json.load(fh)

words = data['words']
categories = data['categories']
train_x = data['train_x']
train_y = data['train_y']

# reset tensorflow default graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model with tensorboard logging and load trained model.
model = tflearn.DNN(net, tensorboard_dir='chatbot_logs')
model.load('./chat_model.tfl')


def bag_array(sentance, words):
    sentance_words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentance)]
    bag = [0] * len(words)

    for s in sentance_words:
        for i, w in enumerate(words):
            if s == w:
                bag[i] = 1

    return(np.array(bag))


def probability(sentence, error_threshold=0.5):
    results = model.predict([bag_array(sentence, words)])[0]

    res = []
    for i, r in enumerate(results):
        res.append((categories[i], r))
        res.sort(key=lambda x: x[1], reverse=True)
    for r in res:
        print(r[0], r[1])

    results = [[i, r] for i, r in enumerate(results) if r > error_threshold]

    # Sort by highest probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append((categories[r[0]], r[1]))

    return return_list


def response(sentence, context, user='123', blackboard_ok=True):
    results = probability(sentence)
    response = ""
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'blackboard_related' in i and not blackboard_ok:
                        response += """
                            Blackboard is currently unavailable.
                            If your query is related to blackboard services
                            please check back later.\n\n"""

                        context[user] = 'blackboard_down_continue'

                    if 'context_set' in i:
                        context[user] = i['context_set']

                    if 'context_filter' not in i or (
                            user in context and 'context_filter' in i and i['context_filter'] == context[user]):
                        response += random.choice(i['responses'])
                        return (response, context)

            results.pop(0)
    else:
        return ("I'm sorry, I didn't quite understand that. Please try rephrasing the question.", context)

print('Hi i have a problem')
print(response('Hi i have a problem', {}), end='\n\n')
print('i cant get on blackboard')
print(response('i cant get on blackboard', {}), end='\n\n')
print('dunno wot my pasword is')
print(response('dunno wot my pasword is', {}), end='\n\n')
print('my shit isnt working')
print(response('my shit isnt working', {}), end='\n\n')
print('yeah mate thats done it cheers')
print(response('yeah mate thats done it cheers', {}), end='\n\n')

while True:
    ctx = {}
    q = input('query: ')
    r, ctx = response(q, ctx)
    print(r)
