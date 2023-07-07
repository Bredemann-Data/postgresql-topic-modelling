#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import psycopg2 as pg
import pandas as pd
import numpy as np

#import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline

from from_sql import from_sql
        
#%% Daten laden

books = from_sql()

table = "book_data"

df = books.query_to_df(table, "true","Title", "ID", "description", "categories")
df = df[df["description"].notna()]

#%%

top_20 = df[2].value_counts().head(30)
print(top_20)

#%% Filter English sentences

from langdetect import detect

def lang_rec(text):
    try:
        lang = detect(text)
    except: 
        return "unknown language"
    else:
        return lang

def add_lang_column(df, spalte, path):
    language = df[spalte].map(lang_rec)
    pickle.dump(language, open(path + '.p', "wb"))
    df['lang'] = language
    return df

df_en = add_lang_column(df=df, spalte="description", path='language')

# check statistics
df_en.value_counts()
#%%

books.update_table_from_df(table= table, columns= ["language"], df= df_en)


#%% download entries that are English and for which there is no topic

condition2 = '"language" = \'en\' and "topic" IS NULL' 

df_en = books.query_to_df(table= table, columns = ["Title", "ID", "description", "categories"], condition = condition2)

#%% combine columns
df_en = df_en.fillna(" ")

# remove brackets
df_en["categories"]  = df_en["categories"].map(lambda x: x[1:-1])

df_en["combined"] = df_en["Tite"] + ' ' + df_en["description"] + ' ' + df_en["categories"]

#%% set length

length = df_en["combined"].map(lambda x: len(x.split()))
length.value_counts()

filter_ = length > 30

longer_summaries = df_en[filter_]

#%% Stop words
import spacy
nlp = spacy.load("en_core_web_md")

from spacy.lang.en import stop_words

stops = list(stop_words.STOP_WORDS)

stops = stops + ['ll', 've', 'book', 'new', 'fiction', 'novel']

#%% switch to lemmas

def lemmatizer(summary):
    doc = nlp(summary)
    lemmas_list = [token.lemma_ for token in doc]
    return " ".join(lemmas_list) 

def series_lemmatizer(series):
    x = series.map(lemmatizer)
    pickle.dump(x, open('lemmatized.p', 'wb'))
    return x

# time needed for lemmatization
import time
import random

def time_estimator():
    estimations = []
    for x in range(10):
        ind = random.randint(0, len(longer_summaries))
        summary = longer_summaries['combined'].iloc[ind]
        start = time.time()
        lemmatizer(summary)
        end = time.time()
        z = end - start
        length_ind = len(summary.split())
        # Zeit pro Zeile
        z_adj = (z / length_ind) * length.mean()
        hours = (len(df_en) * z_adj) / 3600
        estimations.append(hours)
    t = np.array(estimations).mean()
    print(estimations)
    print(f"\nEstimated time for lemmatzation is {t} hours", )

time_estimator()

#%% 

# lemmatized = series_lemmatizer(df_en["combined"])

# This function is noot executed, because lemmatization takes to much time.

#%% topic modelling with LDA
def topic_modeller(data, n_topics, file_name):
    vec = CountVectorizer(stop_words=stops, max_df=0.95, min_df=2)
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=13, n_jobs= -1 
                                    )
    pipe = make_pipeline(vec, LDA)
    # train pipeline
    pipe.fit(data)
    # dump pipeline
    pickle.dump(pipe, open(file_name + ".p", "wb"))
    return pipe

# #%% calclate perplexity

# def get_perplexity(data, n_topics, iters):
#     vec = CountVectorizer(stop_words=stops, max_df=0.95, min_df=2)
#     LDA = LatentDirichletAllocation(n_components=n_topics, random_state=13, n_jobs= -1, 
#                                     max_iter = iters, learning_method="online")
#     pipe = make_pipeline(vec, LDA)
#     # train pipeline
#     pipe.fit(data)
#     LDA = pipe.steps[1][1]
#     vec = pipe.steps[0][1]
#     data = vec.transform(X)
#     perplex = LDA.perplexity(data)
#     print(f"perplexity of model with {n_topics}: {perplex}")

# get_perplexity(longer_summaries['combined'], 27, 20)
# get_perplexity(longer_summaries['combined'], 27, 10)
# print()


#%% modell trainieren

X = longer_summaries['combined']

pipe = topic_modeller(X, 27, "pipe")

#%% Themen herausfinden
#pipe = pickle.load(open('pipe.p', 'rb'))
LDA = pipe.steps[1][1]
vec = pipe.steps[0][1]

topics = LDA.components_


def get_best(k_best, n):
    for index, topic in enumerate(topics):
        if index in n:
            print(f'die häufigsten Wörter in {index}')
            print([vec.get_feature_names_out()[i] for i in topic.argsort()[-k_best:]])
            print()
        else:
            continue

get_best(20, range(len(topics)))


#%% Themen benennen

get_best(30, [17, 16, 13, 11, 10, 4])
get_best(40, [17, 2])

#%% transform data

topic_names = {
    0: 'children, education',
    1: 'American history',
    2: 'other',
    3: 'social',
    4: 'religion',
    5: 'autobiography',
    6: 'mindfulness and selfcare',
    7: 'films, television',
    8: 'business',
    9: 'pets',
    10: 'art, design, architecture',
    11: 'business',
    12: 'computer science',
    13: 'teaching',
    14: 'family',
    15: 'second world war',
    16: 'fantasy',
    17: 'American history',
    18: 'other',
    19: 'health',
    20: 'music',
    21: 'science and technology',
    22: 'crime',
    23: 'travelling',
    24: 'history',
    25: 'poetry',
    26: 'sports'
    }


def get_pred(df, column):
    X = df[column]
    pred = pipe.transform(X)
    X_trans = np.argmax(pred, axis=1)
    df['topic_num'] = X_trans
    df['topic_name'] = df['topic_num'].map(topic_names)
    return df
    
topics = get_pred(df_en, 'combined')

#%% neuen datensatz hochladen

books.write_df_to_server(table= "topics_2", columns= ["ID", "topic"], df = topics[['ID', 'topic_name']])
books.update_table_from_df(table = table, columns="topic", df=topics)
