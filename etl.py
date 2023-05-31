

#

#import des biliotheques
import tweepy

from nltk.corpus import stopwords
import spacy

import pandas as pd
import re
from sqlalchemy import create_engine
import urllib
import warnings
from sqlalchemy.sql import text

pd.options.display.max_colwidth=200
pd.set_option('display.max_columns', None)
#les mots de recherche
search_words=["hyundai","mercedes","bmw"]
#mots reliès aux mots de recherche
related_words=["benz"]
warnings.filterwarnings("ignore")


def clean(item):
    clean_tweet = re.sub("@[A-Za-z0-9_]+","", item) # eliminer les mentions ex @tesla..
    item = re.sub("#[A-Za-z0-9_]+","", clean_tweet) # eliminer les hashtags
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    item=item.lower() #convertir en miniscule
    item=emoji_pattern.sub(r'',item) #eliminer les emoji
    item=re.sub(r'http\S+', '',item) #eliminer les liens
    item=re.sub('\n', '',item) ##eliminer les retour a la ligne
    item=re.sub('$', '',item) #eliminer ces symboles
    item=re.sub('#', '',item)
    item = re.sub('[?!@#$]', '', item)
    for i in search_words:
        item = re.sub(i, '', item) #eliminer les mots de recherche
    for i in related_words:
        item = re.sub(i, '', item) #eliminer les mots reliés
    return(item)



server = 'DESKTOP-QPPVGFN' # to specify an alternate port
database = '' 
username = '' 
password = ''

params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) #connection base de données sql server



#les identifients de connection (reliès au compte twitter developpers)
API_key=""
API_key_secret=""
Bearer_Token=""
access_token=""
access_token_secret=""
Client_ID=""
Client_Secret=""


#connection authorisés

auth=tweepy.AppAuthHandler(API_key,API_key_secret)
api=tweepy.API(auth,wait_on_rate_limit=True)
#liste de toutes les colonnes du fichires json extraites
l=['created_at', 'id', 'id_str', 'full_text', 'truncated', 'display_text_range', 'entities', 'extended_entities', 'metadata', 'source', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'retweeted_status', 'is_quote_status', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive', 'lang']
#dataframe vide contenant ces colonnes
df=pd.DataFrame(columns=l)

#selectionner les id des tweets deja dans la base de données(pour ne pas dupliquer les lignes deja existantes dans la table) 
df5=pd.read_sql(text("select id from tweets"),engine.connect())

ids=list(df5["id"])

for search_word in search_words: # traitement sur chaque mot de recherche
    df1=pd.DataFrame(columns=df.columns[1:]) #df contient la premire colonne "index" à supprimer
    data=tweepy.Cursor(api.search_tweets,q=search_word,tweet_mode="extended",lang='en').items(800) #extraction de 800 tweets en anglais
    for tweet in data:
        if tweet._json['in_reply_to_status_id_str']!=None: # selectionner les tweets qui sont des commentaires
            if 'retweeted_status' in tweet._json:  #j'ai remarqué que si le json contient l' element retweeted_status, le full_text existe seulement dans cet element(merci de voir la structure de json)
                #print(tweet._json['retweeted_status']["full_text"]) 
                df1=df1.append(tweet._json['retweeted_status'],ignore_index=True) 
            else:
                #print(tweet._json["full_text"])
                df1=df1.append(tweet._json,ignore_index=True)
    df2=df1.drop(["user","coordinates","place",'metadata','entities','extended_entities','id_str','display_text_range','source','in_reply_to_status_id_str','in_reply_to_user_id_str','lang','favorited','retweeted'],axis=1)# colonnes non utilisés
    if 'quoted_status' in df2.columns: #eliminer cette colonne si elle existe 
        df2=df2.drop(['quoted_status','quoted_status_id','quoted_status_id_str'],axis=1)
        
    df2["brand_id"]=search_word # enregistrer la marque
    df2["full_text"]=df2["full_text"].apply(lambda x:clean(x)) #appliquer la fonction clean sur chaque texte puis le sauvgarder
    #Validation du cle primaire( eliminer les tweets deja existantes dans la base )
    df2=df2[df2["id"].isin(ids)==False]
    df2.to_sql('tweets', engine, if_exists='append', index = False)



####################################################################################################################################################################


import tweepy

import pandas as pd
import re
from sqlalchemy import create_engine
import urllib
import warnings

pd.options.display.max_colwidth=200


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
#from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from tensorflow import keras
#import torch
#from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
#from sklearn.metrics import accuracy_score, precision_recall_fscore_support






nltk.download("vader_lexicon") #vader lexicon est une dictionnaire des mots positives et negative pour faire une classification des sentiments




df5=pd.read_sql(text("select * from tweets"),engine.connect())




analyzer = SentimentIntensityAnalyzer()

def vader_sentiment_result(sent):
    scores = analyzer.polarity_scores(sent)
    #classer le sentiment selon le score vader
    if scores["compound"] > 0.2:
        return "positive"
    elif scores["compound"] < -0.2:
        return "negative"
    else:
        return "neutral"  
    
df5["full_text"]=df5["full_text"].apply(lambda x:clean(x)) #nettoyer la dataframe avant de traitement du texte

# stop_words et spacy contiennent les prepositions,les pronoms ,les verbes très communs (to be, to do ...) et autres mots communes à supprimer de dataframe
stop_words = set(stopwords.words('english'))  
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
#concatenation de tous ces mots et addition des mots de recherche et mots reliés 
stop_words=list(set(stop_words).union(sw_spacy)) #
stop_words+=search_words
stop_words+=related_words

# supprimer ces mots du dataframe

df5['full_text_stop'] =df5['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#appliquer l'analyse des sentiments
df5["sentiment"] = df5['full_text_stop'].apply(lambda x: vader_sentiment_result(x))




#print(df5[df5["sentiment"]=="negative"])





#print(df5[df5["sentiment"]=="positive"])





#print(df5[df5["sentiment"]=="neutral"])




#enregistrer dans la base de données
df5.to_sql('tweets1', engine, if_exists='replace', index = False)





















