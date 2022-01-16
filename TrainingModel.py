# NAME1:Amat Alaleem Hazbar ID: 20174083
# NAME2:Mustafa Haroun      ID:20161213
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


# Calling the constrcutor Lemmatize using WordNet's built-in function. 
# It returns the input word unchanged if it cannot be found in WordNet.
Lemmatizer = WordNetLemmatizer()

#We have added the responses and the intents in the targets.json file
# We have used a .loads function to load the file to the traning model
# We used a .read function to read our targets file. 
Intents = json.loads(open('targets.json').read())

#define three arrays/lists 
Words = [] #in this array we will store the splited words from the targets.json file.
Classes = [] 
documents = [] #store the word list with thier associated tags.
ignoreSymbols = ["?","!",".",",",";"] # ignore undesirable user's input 

# Here we have used "for" loop,to loop through all the data(intents object) we got from the targets.json file
for intent in Intents['intents']:
    #loop through "patterns" inside the intents object in the targets.json file. 
    for pattern in intent['patterns']:
        #split up the words using a ".word_tokenize" function from  nltk Library. 
        wordList = nltk.word_tokenize(pattern)
        #add combinations of words into our Words list
        Words.extend(wordList)
        #add word list into documents with the tag of that word.
        documents.append((wordList,intent['tag']))
        #check if this category is not already in the classes.
        if intent['tag'] not in Classes:
            Classes.append(intent['tag'])
 
#check if a word is  in the  list of ignored characters.
Words = [Lemmatizer.lemmatize(word) for word in Words if word not in ignoreSymbols]
#Remove duplicate words
Words = sorted(set(Words))
#Remove duplicate classes
Classes = sorted(set(Classes))

#save words and classes into a pickle files.
# the files will be aurtomatically generated when the tranining model run.
pickle.dump(Words,open('WordStorage.pkl','wb'))
pickle.dump(Classes,open('ClasseStorage.pkl','wb'))

#To train the neural network we have to represent words as numerical values
trainingData = []
empty_output = [0] * len(Classes)

for document in documents:
    word_bag = []
    pattern_words = document[0]
    pattern_words = [Lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in Words:
        word_bag.append(1) if word in pattern_words else word_bag.append(0)

    row_output = list(empty_output)
    row_output[Classes.index(document[1])] = 1
    trainingData.append([word_bag,row_output])

#randomize data
random.shuffle(trainingData)
trainingData = np.array(trainingData)
#split into X and Y values
trainX = list(trainingData[:,0])
trainY = list(trainingData[:,1])

#build neural network
model = Sequential()
model.add(Dense(128,input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation = 'softmax'))

#optimize by adding learning rate.
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_model.h5',hist)
print("done")







