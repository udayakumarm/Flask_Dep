import numpy as np
import pandas as pd
import re

import seaborn as sns
#import matplotlib.pyplot as plt
#%matplotlib inline

import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from imblearn.over_sampling import SMOTE
#from collections import Counter
#from sklearn.metrics import roc_auc_score,roc_curve
#from xgboost import XGBClassifier
#from sklearn.naive_bayes import GaussianNB
#from spellchecker import SpellChecker
#from symspellpy import SymSpell
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.model_selection import train_test_split

import joblib
from nltk.stem import WordNetLemmatizer

def get_first_val(group):
    total_reviews=len(group.sent_pred)
    total_1s=(group['sent_pred'] == 1).sum()
    postive_pre=((total_1s*100)/total_reviews)
    return(postive_pre)

def myfuc_PR(name_inp):
    df_org = pd.read_csv("sample30.csv")
    df_org_train, df_org_test = train_test_split(df_org, test_size=0.30, random_state=31)

    #df_org_train.head
    df_org_train["review_title_text"] = df_org_train["reviews_title"] + df_org_train["reviews_text"]

    df_pivot = df_org_train.pivot_table(
        index='reviews_username',
        columns='id',
        values='reviews_rating'
    ).fillna(0)

    #df_pivot.head(100)

    dummy_train = df_org_train.copy()

    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot_table(
        index='reviews_username',
        columns='id',
        values='reviews_rating'
    ).fillna(1)

    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T-mean).T
    df_subtracted.head()


    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0

    user_correlation[user_correlation<0]=0

    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

    #user_input = input("Enter your user name")
    user_input = name_inp
    print(user_input)

    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    d = pd.merge(d,df_org,left_on='id',right_on='id', how = 'left')

    d["review_title_text"] = d["reviews_title"] + d["reviews_text"]

    #RandFclassifer = joblib.load("model-RF.pkl")
    #RandFclassifer = joblib.load("model-XGBoost.pkl")
    RandFclassifer = joblib.load("model-LR.pkl")


    X = d['review_title_text']


    documents = []


    stemmer = WordNetLemmatizer()
    #sym_spell = SpellChecker()

    for sen in range(0, len(X)):
      
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
    
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()
    
        # Lemmatization
        document = document.split()
     
        #lemmatization
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
    
        documents.append(document)

    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(documents).toarray()

    y_sent=RandFclassifer.predict(X)

    d['sent_pred']=y_sent

    final_list= d.groupby('id').apply(get_first_val).dropna()
    FL = (final_list.sort_values(ascending=[False]).head(5))
    return (FL.index.tolist())






