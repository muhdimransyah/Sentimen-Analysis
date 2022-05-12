# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:17:04 2022

@author: imran
"""

import os
import re
import datetime
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

a = 'I am a boy 12345'
print(a.replace('1234567',''))
print(re.sub('[^a-zA-Z]','',a).lower())

#%% 
# EDA
# Data Loading

URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
LOG_PATH = os.path.join(os.getcwd(), 'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
df = pd.read_csv(URL)

review = df['review']
review_dummy = review.copy() # X_train

sentiment = df['sentiment']
sentiment_dummy = sentiment.copy() # Y_train

# Data Inspection
review_dummy[3]
sentiment_dummy[3]

review_dummy[11]
sentiment_dummy[11]

print(review_dummy[11].replace('<br />',''))

# Data Cleaning
# Remove html tags
for index, text in enumerate(review_dummy):
    #review_dummy[index] = text.replace('<br />','')
    review_dummy[index] = re.sub('<.*?>', '', text)# .*Everything inside the < > will be filtered
    
# convert to lowercase and split it
for index, text in enumerate(review_dummy):
    review_dummy[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()

# Data Vectorization for Reviews
num_words = 10000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(review_dummy)

# To save the tokenizer for deployment purpose
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(),'tokenizer.data.json')
token_json = tokenizer.to_json()

with open(TOKENIZER_JSON_PATH,'w') as json_file:
    json.dump(token_json,json_file)

# To observe number of words
word_index = tokenizer.word_index
print(word_index)
print(dict(list(word_index.items())[0:10]))

# To vectorize the sequences of the text
review_dummy = tokenizer.texts_to_sequences(review_dummy)
#pad_sequences(review_dummy, maxlen=200)

temp = [np.shape(i) for i in review_dummy] # to check number words inside list
np.mean(temp) #mean num of words --> 234
review_dummy = pad_sequences(review_dummy,
                             maxlen=200,
                             padding='post',
                             truncating='post')

# For label
one_hot_encoder = OneHotEncoder(sparse=False)
sentiment_encoded = one_hot_encoder.fit_transform(np.expand_dims(sentiment_dummy,axis=-1))

X_train, X_test, y_train, y_test = train_test_split(review_dummy, 
                                                    sentiment_encoded, 
                                                    test_size=0.3, 
                                                    random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
 
one_hot_encoder.inverse_transform(np.expand_dims(y_train[0],axis=-1))
# positive is [0,1]
# negative is [1,0]

#%% Model Creation

model = Sequential()
model.add(Embedding(num_words, 64))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.summary()

#model = Sequential()
#model.add(LSTM(128, input_shape=(np.shape(X_train)[1:]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(128))
#model.add(Dropout(0.2))
#model.add(Dense(2,activation='softmax'))
#model.summary()

#%% Callbacks

log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

#%% Compile and Model Fitting

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='acc')
model.fit(X_train,
          y_train,
          epochs=3,
          validation_data=(X_test,y_test),
          callbacks=tensorboard_callback)

#%% Model Evaluation

#predicted = []
#for i in X_test:
#    predicted.append(model.predict(np.expand_dims(i,axis=0)))
    
predicted_advanced = np.empty([len(X_test), 2])
for index, i in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(i,axis=0))
    
#%% Model Analysis

y_pred = np.argmax(predicted_advanced,axis=1)
y_true = np.argmax(y_test,axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

model.save(MODEL_SAVE_PATH)
