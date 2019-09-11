import pandas as pd
import fasttext
import io
import matplotlib.pyplot as plt
import pandas
import math
import requests
from functools import reduce
from lxml import html
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import spacy
import numpy as np
from keras.models import load_model
model = load_model('model/lever1.keras')  # loads a HDF5 file 'lever1.keras'

#Load spacy
nlp = spacy.load('en_core_web_sm')

#Load wiki trained fasttext word vectors
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(tokens[1:])
    return data

#Forms a word embedding dictionary for searching representations
#These pretrained word embeddings are provided by fasttext. One can also use Glove Vectors or Word2Vec files
wordvectors = load_vectors("fasttext/wiki-news-300d-1M.vec")


#Define NLP functions
#Represent word
class levermodel :
    def __init__(self) :
        pass
    #Searches for vector representation of a word
    def represent_word(self,word) :
        try :
            return np.asarray(wordvectors[word.lower()])
        except :
            return None
        
    #Uses NLP to only find entity that are nouns or adverbs
    def tx2tokens(self,tx) :
        doc = nlp(tx)
        tokens =  list(map(lambda x : x.text, filter(lambda x : x.pos_=="NOUN" or x.pos_=="ADV",doc)))
        #tokens =  list(map(lambda x : x.text, doc))
        return tokens
    
    #Represent a full textual requirement to be consumed by LSTM Sequential layer in form of numpy array of (1,None,300)
    def represent_tx(self,tx) :
        data = np.array(list(filter(lambda x : x is not None,map(lambda x : self.represent_word(x), self.tx2tokens(tx)))))
        return data.reshape(1,data.shape[0],data.shape[1])

    #Runs a inference on the saved model file
    def modelinfer(self,desc) :
        x = self.represent_tx(desc)
        return model.predict(x,batch_size=1)


class parser :
    def __init__(self) :
        pass
    def parseUrl(self,url) :
        BASE_URL = "https://jobs.lever.co/"
        url = BASE_URL+url
        contents = requests.get(url).content
        tree = html.fromstring(contents)
        #title = tree.xpath('/html/body/div[2]/div/div[1]/div/div[1]/h2/text()')
        title = tree.xpath('//div[@class="posting-headline"]/h2/text()')
        requirements_node = tree.xpath('//div[@class="section page-centered"]')
        requirements = reduce(lambda x,y : x +" "+y, map(lambda x :''.join(x.itertext()),requirements_node))
        return [title[0], requirements]

from flask import Flask 
app = Flask(__name__)
m = levermodel()
p = parser()
@app.route('/predict/salary/<path:uniqueid>')
def find_expected_salary(uniqueid):
    print("Predicting....")
    try :
        sal = str(round(m.modelinfer(data[1])[0][0],2))
        return "{"+"'Title:'"+data[0]+","+"'Predicted':"+sal+"}"
    except Error as error :
        print(error)
        return "{"+"'Error':"+"Please check the level id"+"}"

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
