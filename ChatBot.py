# NAME1:Amat Alaleem Hazbar ID: 20174083
# NAME2:Mustafa Haroun      ID:20161213

import random
import json
import pickle
import numpy as np
import nltk
#download missing modules of the nltk library. 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
Lemmatizer = WordNetLemmatizer()
# load and read the content of the targets.json file 
Intents = json.loads(open('targets.json').read())

# open the pickle file which has been created by the training model.
words = pickle.load(open('WordStorage.pkl','rb'))
classes = pickle.load(open('ClasseStorage.pkl','rb'))
model = load_model('chatbot_model.h5')

# create a function to clean the sentences up by using . word_tokenize function.
def clean_sent(sent):
    wordsents = nltk.word_tokenize(sent)
    wordsents = [Lemmatizer.lemmatize(word) for word in wordsents]
    # return a list of words .
    return wordsents

#this function will create Storage of the words
def storage_of_words(sent):
    wordsents = clean_sent(sent)
    bag = [0] * len(words) # the size of hte storage.
    for w in wordsents:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 

    return np.array(bag)

# fucntion to predict the classes of the sentences
def classesPredict(sent):
    BOW = storage_of_words(sent)
    res = model.predict(np.array([BOW]))[0]
    Error = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > Error]
    results.sort(key = lambda x:x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

# function of the responses.
def response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result 

print("ChatBot:Hi there!My name is Alice, Covid-19 helpdesk assistant.I can help you recognize your symptoms and guide you on what to do if you have Covid via Chat.")

while True:
    # get input from the user.
    message = input("User:")
    #call the classespredic function and pass the user message as a parameter.
    ints = classesPredict(message)
    # call the response function to give the user an answer dependends on in which
    # class his question is classified.
    res = response(ints,Intents)
    print("ChatBot:"+res)

