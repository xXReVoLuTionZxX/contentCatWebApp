#!/usr/bin/env python
# coding: utf-8

from flask import Flask, redirect, url_for, render_template, request
from dotenv import load_dotenv

import os
import tweepy
import webbrowser
import time
import pandas as pd
import numpy as np 
import tensorflow as tf
import urllib.request
from transformers import BertTokenizer

load_dotenv(dotenv_path="./.env.local")

app = Flask(__name__)

model = tf.keras.models.load_model('CapstoneBertModel')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') #bert tokenizer 
def prep_data(text):
    tokens = tokenizer.encode_plus(text, max_length=512, #seq leng is 512 is more than that will be truncated and if less the remain will be padded 
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False, #special tokens will be added
                                   return_tensors='tf') #retun tensor
    
    return {
        'input_ids': tf.cast(tokens['input_ids'], tf.float64), 
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
    }

CONSUMER_KEY = os.environ.get("CONSUMER_KEY", "") #developer key twitter
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET", "") #developer key secret twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET) #returns the object with the auth
ACCESS_TOKEN= os.environ.get("ACCESS_TOKEN", "") #access token
ACCESS_TOKEN_SECRET= os.environ.get("ACCESS_TOKEN_SECRET", "")#access token secret 
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET) #get the authentication and access
api = tweepy.API(auth)

@app.route("/") #define home route
def home():
    return render_template('home.html') #return to home.html

@app.route("/description") #define description route
def description():
    return render_template('description.html') #return to description.html

@app.route("/faq") #define faq route
def faq():
    return render_template('faq.html') #return to faq.html 

@app.route("/no_size") #define nosize route 
def no_size():
    return render_template('no_size.html') #return to no_size.html

@app.route("/result", methods=['POST']) #define result route using method post 
def result():

    x= time.time() #returns the time as floating point number expressed in seconds in UTC 
    username = request.form["username-input"] #get the username that the user input from the textbox.
    userwithoutspaces = username.replace(" ", "") #remove spaces
    user = userwithoutspaces.isalnum() #only accept a-z and 0-9
    if user == True:
        userID = userwithoutspaces #if the userID is valid then will the username is accept
    else:
        return render_template('no_user.html') #if the username is not valid the user will be redirect to no_user.html

    try:    
        tweets = api.user_timeline(screen_name=userID, 
                            # 200 is the maximum allowed count
                            count=200,
                            include_rts = False, #retweets will not appear
                             exclude_replies = True, #replies will appear
                            tweet_mode = 'extended'#keep tweets full text
                            ) #get user tweets
        
        #create a table with the following columns user author and tweets
        columns=set()
        allowed_types = [str, int]
        tweets_data = []
        for status in tweets:
            status_dict = dict(vars(status))
            keys = status_dict.keys()
            single_tweet_data = {"user": status.user.screen_name, "author": status.author.screen_name}
            for k in keys:
                try:
                    v_type = type(status_dict[k])
                except:
                 v_type = None
                if v_type != None:
                    if v_type in allowed_types:
                        single_tweet_data[k] = status_dict[k]
                        columns.add(k)
        tweets_data.append(single_tweet_data)
        header_cols = list(columns)
        header_cols.append("user")
        header_cols.append("author")
        df = pd.DataFrame(tweets_data, columns=header_cols) #a table filled with the information is create
        tweets_text = df.full_text #only takes the tweets from the tweets column. 
        test = prep_data(f'{tweets_text}') #tokenize the test 
        probs = model.predict(test) #predicts the political bias
        prediction = np.argmax(probs[0])#returns the index of the maximum value
        filename="./static/ProfilePicture/user.jpg" #assign the path in which the picture will be save.
        path=status.user.profile_image_url_https #store the url of the profile picture of user.
        path = path.replace('_normal',"") #to get picture full size _normal word is deleted
        urllib.request.urlretrieve(path,filename) #request an store the image 
        if prediction == 0: 
            return render_template('result-left.html', data=userID, x=x) #if the user has a left prediction will be redirect to resutl-left.html.
        elif prediction == 1:
            return render_template('result-right.html', data=userID, x=x) #if the user has a right prediction will be redirect to result-right.html.
        else:
            return render_template('result-center.html', data=userID, x=x) #if the user has center prediction will be redirect to result-center.html.
    except:
            return render_template('no_user.html') #if any error occur the user will be redirect to no_user.html.


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80) 
