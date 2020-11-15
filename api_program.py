import eazyml
import pandas as pd
import numpy as np

#to start the program, make sure you change the last number 
#in the second row to the id of the county you which to check

#authentication to use api
username = 'adsarwaikar@ctemc.org'
password = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkNmIxOTNhNS05ZDAxLTRlYzUtOTZhNC0zZmU0NDNkYWE2NjciLCJleHAiOjE2MDU0Njg4NzQsImZyZXNoIjpmYWxzZSwiaWF0IjoxNjA1MzgyNDc0LCJ0eXBlIjoiYWNjZXNzIiwibmJmIjoxNjA1MzgyNDc0LCJpZGVudGl0eSI6IkFkaXR5YSBTYXJ3YWlrYXIifQ.0-q6NHMxmhCQu3u-XVVzLASb_AzxtpS4VxI78M6AMIU'
train_file_path = 'dataset.csv'

resp = eazyml.ez_auth(username, None, password)
auth_token = resp["token"]

options = {
    "id": "ID",
    "impute": "yes",
    "outlier": "yes",
    "discard": "null",
    "accelerate": "yes",
    "outcome" : "Major Incident"
}

ez_model_config = {
    "model_type" : "predictive",
    "derive_text" : "no",
    "derive_numeric": "no",
    "accelerate": "yes"
}

#loading the training data
resp = eazyml.ez_load(auth_token, train_file_path, options)
dataset_id = resp["dataset_id"]

#building the model
resp = eazyml.ez_init_model(auth_token, dataset_id, ez_model_config)
model_id = resp["model_id"] 
model_name = resp["model_performance"]["data"][0][0]

options = {
    "model_name": model_name
}

#getting final response with answers and displaying to user
response = eazyml.ez_predict(auth_token, model_id, 'prediction.csv', options)
for row in response['predictions']['data']:
    answer = row[-1]
    if answer == 'TRUE':
        print("There is a likely chance that a wildfire here would become a major incident. ")
    if answer == 'FALSE':
        print("There is an unlikely chance that a wildfire here would become a major incident. ")