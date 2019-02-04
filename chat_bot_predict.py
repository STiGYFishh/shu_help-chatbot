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
    results = [[i, r] for i, r in enumerate(results) if r > error_threshold]

    # Sort by highest probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append((categories[r[0]], r[1]))

    return return_list


def response(sentence, context, blackboard_ok=True):
    manual_intervention = (
        'I seem to be having an issue understanding the problem, '
        'you can contact SHU IT Support directly by emailing ithelp@shu.ac.uk,'
        ' or by phoning 0114 225 3333.'
        '\nIf you do not wish to contact IT Support directly you can try rephrasing the question.')

    try:
        if context['not_understood'] >= 3:
            context['not_understood'] = 0
            return (manual_intervention, context)
    except KeyError:
        context['not_understood'] = 0

    results = probability(sentence)
    response = ""
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        context['path'] = i['context_set']

                    if 'blackboard_related' in i and not blackboard_ok:
                        context['blackboard'] = True

                    if 'context_filter' not in i or i['context_filter'] == context['path']:
                        response = random.choice(i['responses'])

                    return (response, context)

            results.pop(0)
    else:
        context['not_understood'] += 1
        return ("I'm sorry, I didn't quite understand that. Please try rephrasing the question.\n", context)
