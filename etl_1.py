#import des biliotheques
import tweepy



import pandas as pd
import re
from sqlalchemy import create_engine
import urllib
import warnings
from sqlalchemy.sql import text

#les mots de recherche
search_words=["hyundai","mercedes","bmw"]
#mots reliès aux mots de recherche
related_words=["benz"]
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
nb_tweets=800

    
def clean(item):
    clean_tweet = re.sub("@[A-Za-z0-9_]+","", item) # eliminer les mentions ex @tesla..
    item = re.sub("#[A-Za-z0-9_]+","", clean_tweet) # eliminer les hashtags
    emoji_pattern = re.compile("["
        
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    item=item.lower() #convertir en miniscule
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



server = '' # to specify an alternate port
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
l=['created_at', 'id', 'id_str', 'full_text', 'truncated', 'display_text_range', 'entities', 'extended_entities', 'metadata', 'source', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'retweeted_status', 'is_quote_status', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive', 'lang','brand_id']
#dataframe vide contenant ces colonnes
df=pd.DataFrame(columns=l)

#selectionner les id des tweets deja dans la base de données(pour ne pas dupliquer les lignes deja existantes dans la table) 
df5=pd.read_sql(text("select id from tweets"),engine.connect())

ids=list(df5["id"])

df1=pd.DataFrame(columns=df.columns[1:])

for search_word in search_words: # traitement sur chaque mot de recherche
     #df contient la premire colonne "index" à supprimer
    data=tweepy.Cursor(api.search_tweets,q=search_word,tweet_mode="extended").items(nb_tweets) #extraction de 800 tweets en anglais
    for tweet in data:
        #if tweet._json['in_reply_to_status_id_str']!=None: # selectionner les tweets qui sont des commentaires
            if 'retweeted_status' in tweet._json:  #j'ai remarqué que si le json contient l' element retweeted_status, le full_text existe seulement dans cet element(merci de voir la structure de json)
                print(tweet._json['retweeted_status']["full_text"]) 
                df1=df1.append(tweet._json['retweeted_status'],ignore_index=True) 
            else:
                print(tweet._json["full_text"])
                df1=df1.append(tweet._json,ignore_index=True)
            i=search_words.index(search_word)
            df1.loc[i*nb_tweets:(i+1)*nb_tweets,"brand_id"]=search_words[i]

                

    
df2=df1 [['created_at', 'id', 'full_text',
       'in_reply_to_status_id', 'in_reply_to_user_id',
       'in_reply_to_screen_name', 
       'retweeted_status', 'retweet_count',
       'favorite_count', 'possibly_sensitive',"brand_id","lang"]]
df2.drop_duplicates("id",inplace=True)


     # enregistrer la marque
    
df2["full_text"]=df2["full_text"].apply(lambda x:clean(x)) #appliquer la fonction clean sur chaque texte puis le sauvgarder
    #Validation du cle primaire( eliminer les tweets deja existantes dans la base )
    
df2=df2[df2["id"].isin(ids)==False]
df2[['created_at','full_text','in_reply_to_screen_name','retweeted_status']] = df2[['created_at','full_text','in_reply_to_screen_name','retweeted_status']].astype(str)
df2.to_sql('tweets', engine, if_exists='append', index = False)



####################################################################################################################################################################


import tweepy

import pandas as pd
import re
from sqlalchemy import create_engine
import urllib
import warnings

pd.options.display.max_colwidth=200




MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)





def sentiment_scores(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return (scores)


def sentiment_result(text):
    scores = sentiment_scores(text)
    #classer le sentiment selon le score 
    if scores[0]==scores.max():
        return "negative"
    elif scores[2]==scores.max(): 
        return "positive"
    else:
        return "neutral"
    
    
    
locations =pd.read_csv("locations.csv")
locations=list(locations.to_numpy().reshape(-1,))
locations = [str(x) for x in locations]


def location_check(text):
    if any((match :=ext) in text.lower() for ext in locations):
        return match
    else:
        return "False"
    
    
    
    
    
df2["full_text"]=df2["full_text"].apply(lambda x:clean(x)) #nettoyer la dataframe avant de traitement du texte



#appliquer l'analyse des sentiments
df2["sentiment"]=" "
df2.loc[df2['lang']=="en","sentiment"] = df2.loc[df2['lang']=="en",'full_text'].apply(lambda x: sentiment_result(x))

df2["location"]=pd.DataFrame(list(df1['user']))["location"]
df2["location_check"]=df2["location"].apply(lambda x:location_check(x))





#enregistrer dans la base de données
df2.to_sql('tweets1', engine, if_exists='append', index = False)
df2.to_sql('tweets2', engine, if_exists='append', index = False)
