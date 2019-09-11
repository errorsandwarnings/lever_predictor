import pandas as pd
import fasttext
import io
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import spacy
import numpy as np

#Load spacy
nlp = spacy.load('en_core_web_sm')

#Load wiki trained word vectors
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(tokens[1:])
    return data

wordvectors = load_vectors("../fasttext/wiki-news-300d-1M.vec")


#Load kaggle training dataset
dataset = pd.read_csv("../data/Train_rev1.csv")
dataset = dataset[["Title","FullDescription","Category","SalaryNormalized"]]

#Define NLP functions
def represent_word(word) :
    try :
        return np.asarray(wordvectors[word.lower()])
    except :
        return None

def tokenize(sentence) :
    tokens = word_tokenize(sentence)
    nltk.pos_tag(tokens)
    
def tx2tokens(tx) :
    doc = nlp(tx)
    tokens =  list(map(lambda x : x.text, filter(lambda x : x.pos_=="NOUN" or x.pos_=="ADV",doc)))
    #tokens =  list(map(lambda x : x.text, doc))
    return tokens

def represent_tx(tx) :
    data = np.array(list(filter(lambda x : x is not None,map(lambda x : represent_word(x), tx2tokens(tx)))))
    return data.reshape(1,data.shape[0],data.shape[1])

# create the LSTM network
model = Sequential()
model.add(LSTM(128, input_shape=(None,300),kernel_initializer='he_uniform'))
model.add(Dense(64,activation="relu"))
model.add(Dense(32))
model.add(Dense(1,activation="linear"))
model.compile(loss='mean_squared_logarithmic_error', optimizer="adam", metrics=['mse'])

def modeltrain(desc,sal) :
#Define Training Data
    trainX = represent_tx(desc)
    trainY = np.array([sal])
#Fit the LSTM network
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
    
def modelinfer(desc) :
    x = represent_tx(desc)
    return model.predict(x,batch_size=1)
    
for index, rec in dataset.iterrows():
    desc = rec["FullDescription"]
    sal = rec["SalaryNormalized"]
    print(sal)
    modeltrain(desc,sal)
    
model.save("lever1.keras")
