# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:44:11 2022

@author: imran
"""

import os
import datetime
import numpy as np
import pandas as pd
from sentiment_analysis_module import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
TOKEN_SAVE_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(), 'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review)
review = eda.lower_split(review)

review = eda.sentiment_tokenizer(review, TOKEN_SAVE_PATH)
review = eda.sentiment_pad_sequence(review)

one_hot_encoder = OneHotEncoder(sparse=False)
sentiment = one_hot_encoder.fit_transform(np.expand_dims(sentiment,axis=-1))

nb_cat = len(np.unique(sentiment))

 # X = review, Y = Sentiment
X_train, X_test, y_train, y_test = train_test_split(review, 
                                                    sentiment, 
                                                    test_size=0.3, 
                                                    random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
x_test = np.expand_dims(X_test,axis=-1)

print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0],axis=-1)))

mc = ModelCreation()
num_words = 10000
model = mc.lstm_layer(num_words, nb_cat)

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

predicted_advanced = np.empty([len(X_test), 2])
for index, i in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(i,axis=0))
    
y_pred = np.argmax(predicted_advanced,axis=1)
y_true = np.argmax(y_test,axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

model.save(MODEL_SAVE_PATH)
