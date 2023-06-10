import streamlit as st
from streamlit_lottie import st_lottie
import json
import requests
import pandas as pd
import numpy as np
import pickle
import nltk
import string
import plotly.express as px
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob.sentiments import NaiveBayesAnalyzer

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('white.png')

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'img_1.png'
sidebar_bg(side_bg)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

pickle_in=open('ml_model (3).pkl','rb')
tfidf_in=open('tfidf (2).pkl','rb')
log=pickle.load(pickle_in)
tfidf=pickle.load(tfidf_in)

df=pd.read_excel('hotel_reviews.xlsx')

# Let's change the rating to be more general and easier to understand
def ratings(score):
    if score > 3:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'
df1 = df.copy()
df1['Rating'] = df['Rating'].apply(ratings)


def cleaning(text):
    # remove punctuations and uppercase
    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # remove stopwords
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]

    # lemmatize the word
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)

def ml_predict(text):
    clean_text = cleaning(text)
    tfid_matrix = tfidf.transform([clean_text])
    pred_proba = log.predict_proba(tfid_matrix)
    pred = log.predict(tfid_matrix)[0]

    return pred


menu = ['About','Dataset','Sentiment of your Sentence','Dataset Reviews']
chart = st.sidebar.radio("SELECT THE OPTION:-", menu)

if chart == 'About':
    st.header('Hotel Rating Classification')
    st.subheader('Business Objective:-')
    st.write("T and major objective is what are the attributes that travelers are considering while selecting a hotel. With this manager can understand which elements of their hotel influence more in forming a positive review or improves hotel brand image.")
    lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_JXUInT.json")
    st_lottie(lottie,key="Rating")
    #lottie_cod = load_lottiefile("hotel1.json")
    #st_lottie(lottie_cod)


if chart == 'Dataset':
    st.subheader('Data Information')
    h = ['Show Dataset','jajhjh']
    if st.button('Show Dataset'):
        a = df.head()
        b = df.tail()
        st.write('First five reviews:-',a)
        st.write('Last five reviews:-',b)
        st.write('Shape of the Dataset:-')
        c = df.shape
        st.write(c)

    if st.button('Rating count'):
        st.bar_chart(df1['Rating'].value_counts())



if chart == 'Sentiment of your Sentence':
    text = st.text_input('Enter your Sentence/Review :',"")
    if st.button('Check spellings and Analyse'):
       blob=TextBlob(text) 
       aa = blob.correct()
       st.write('After Spelling Correction:-')
       st.write(aa)
       bb = str(aa)
       blob2 = TextBlob(bb)
       st.write('Sentiment of your review:-')
       st.subheader(ml_predict(bb))
       #st.write('Polarity:-', round(blob2.sentiment.polarity, 2))
       st.write('Subjectivity:-', round(blob2.sentiment.subjectivity, 2))

if chart == 'Dataset Reviews':

    #s = st.slider('enter review number:-',min_value=0,max_value=20490)
    st.subheader('Enter the row number of review:-')
    st.write('Choose between 0 to 20490')
    s = st.number_input('',min_value=0,max_value=20490)
    if st.button('Analyse'):
       text1 = df['Review'][s]
       st.write(text1)
       st.subheader('Sentiment of your review:-')
       st.subheader(ml_predict(text1))
       blob1=TextBlob(text1)
       #st.write('Subjectivity:-',blob1.sentiment.subjectivity)
       st.write('Polarity:-', round(blob1.sentiment.polarity, 2))
       st.write('Subjectivity:-',round(blob1.sentiment.subjectivity, 2))


