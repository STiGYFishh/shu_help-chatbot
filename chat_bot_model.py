import json
import numpy as np
import tensorflow as tf
import tflearn
import random
import nltk
from ntlk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open('intents.json', 'r') as data:
    intents = json.load(data)

word_list = []
categories = []
documents = []

ignore_list = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.tokenize(pattern)
        word_list.extend(word)

        documents.append((word, intent['tag']))

        if intent['tag'] not in categories:
            categories.append(intent['tag'])

words = [stemmer.stem(word.lower()) for word in word_list if word not in ignore_list]

words = sorted(list(set(words)))
categories = sorted(list(set(categories)))

training = []
output = []
empty_output = [0] * len(categories)

for document in documents:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in document[0]]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(empty_output)
    output_row[categories.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset tensorflow default graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model with tensorboard logging
model = tflearn.DNN(net, tensorboard_dir='chatbot_logs')

# Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('chat_model.tfl')

data_structures = {
    'words': words,
    'categories': categories,
    'train_x': train_x,
    'train_y': train_y
}

with open('training_data.json', 'w') as fh:
    json.dump(data_structures, fh, indent=4)
