# %%
from fastapi import FastAPI
from fastapi import Request
import requests
import pandas as pd
import csv 

#%
#function for requesting to the API the use of a model
app = FastAPI()
@app.get("/vision")
def read_vision(img_url):
    url = 'https://api-us.restb.ai/vision/v2/multipredict'
    payload = {
        # Add your client key
        'client_key': 'b1df17d9f440b1a56f07936fb84c56039d4374c86c91618a938e465e8194876a',
        'model_id': ['re_condition_r1r6_global', 're_features_v5'], 
        # Add the image URL you want to process
        'image_url': img_url
    }

    # Make the API request
    response = requests.get(url, params=payload)

    # The response is formatted in JSON
    json_response = response.json()
    return json_response

#%%
#Reading the extracted urls of images
df = pd.read_csv('urls.csv', index_col=0, sep = ';').reset_index()


# %%
#getting a random sample of the df to avoid surpasing the request limit
description_df = df.sample(n=3600, random_state=434143)

# %%
#saving per batches the requests sent to the api
with open('save_descriptions.csv', 'w', newline='') as file:
    for idx in description_df.index:
        descr = read_vision(description_df['Url'][idx])
        file.write(str(descr) + ";")
        if idx%10 == 0:
            file.flush()
file.close()
