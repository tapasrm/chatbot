import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers.legacy import SGD
import random

# Import the Train_bot file for pre-processing

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("Train_Bot.json").read()
intents = json.loads(data_file)

# Pre-processing JSON data
# Tokenization

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # ad documents in the corpus
        documents.append((w, intent['tag']))
        
        # add to Classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize each word and remove duplicates
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
#print("~Document Length~")
#print(len(documents), "documents\n\n", documents)
#print("-"*100)

# classes = intents
#print("~Class Length~")
#print(len(classes), "classes\n\n", classes)
#print("-"*100)

# words = all words, vocabulary
#print("~Word Length~")
#print(len(words), "unique lemmatized words\n\n", words)

#pickle.dump(words, open('words.pkl', 'wb'))
#pickle.dump(classes, open('classes.pkl', 'wb'))

# creating training data
training = []

# create an empty array for output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence 
for doc in documents:
    # initialize our bag of words 
    bag = []
    # list tokenized words for the pattern 
    pattern_words = doc[0]

    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # create our bags of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # 'output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffling features and converting into numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# create train and test list
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")

# creating NN model to predict the responses
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Using SGD with Nesterov accelerated gradient
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose='1')
model.save('chatbot.h5', hist)
print("\n")
print("*"*50)
print("\n Model created Succesfully!")

# load saved model file
model = load_model('chatbot.h5')
intents = json.loads(open("Train_Bot.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):

    #tokenize the patterns and split words into array 
    sentence_words = nltk.word_tokenize(sentence)

    #stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):

    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
               
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
   
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# function to get the response from the model

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# function to predict the class and get the response

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# function to start the chat bot which will continue till the user type 'end'

def start_chat():
    print("Bot: This is Sophie! Your Personal Assistant.\n\n")
    while True:
        inp = str(input()).lower()
        if inp.lower()=="end":
            break
        if inp.lower()== '' or inp.lower()== '*':
            print('Please re-phrase your query!')
            print("-"*50)
        else:
            print(f"Bot: {chatbot_response(inp)}"+'\n')
            print("-"*50)

start_chat()



