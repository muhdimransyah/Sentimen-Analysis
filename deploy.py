# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:20:33 2022

@author: imran
"""

import os, json
import numpy as np
from tensorflow.tools import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sentiment_analysis_module import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')
JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

with open(JSON_PATH,'r') as json_file:
    token = json.load(json_file)
    
new_review = "I think the first one hour is interesting but /the second half of the movie is boring. Time is being wasted a lot. Waste of money."

new_review = [list(input('Review about the movie/n'))]

eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

loaded_tokenizer = tokenizer_from_json(token)
new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequence(new_review)

outcome = sentiment_classifier.predict(np.expand_dims(new_review,axis=-1))

sentiment_dict = {1:'positive', 0:'negative'}
print('This review is ' + sentiment_dict[np.argmax(outcome)])