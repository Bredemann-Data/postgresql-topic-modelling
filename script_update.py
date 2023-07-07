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

#%% Load as df

books = from_sql()

table = "book_data"

condition = '"language" IS NULL' 

df = books.query_to_df(table, condition, "Title", "ID", "description", "categories")
df = df[df["description"].notna()]

#%% Assign languages

from langdetect import detect

def lang_rec(text):
    try:
        lang = detect(text)
    except: 
        return "unknown language"
    else:
        return lang

def add_lang_column(df, spalte):
    language = df[spalte].map(lang_rec)
    #pickle.dump(language, open(path + '.p', "wb"))
    df['language'] = language
    print("languages detected")
    return df

df_lang = add_lang_column(df=df, spalte="description")

#%% update db.table with languages

books.update_table_from_df(table= table, columns= ["language"], df= df_lang)


#%% topic modelling

#%% download entries that are English and for which there is no topic

condition2 = '"language" = \'en\' and "topic" IS NULL' 

df_tm = books.query_to_df(table= table, columns = ["Title", "ID", "description", "categories"], condition = condition2)

#%% remove brackets from category

def column_combiner(df):
    # provide empty strings for N.A
    df = df.fillna(" ")
    # remove brackets
    df["categories"]  = df["categories"].map(lambda x: x[1:-1])
    # column that combines title, description and categories in a single string
    df["combined"] = df["Title"] + ' ' + df["description"] + ' ' + df["categories"]
    
    #set length

    # length = df["combined"].map(lambda x: len(x.split()))

    # filter_ = length > 30

    return df#[filter_]

df_tm = column_combiner(df_tm)
    
#%% download pretrained pipeline

# wd for LDA-pipeline:
wd = "location of pipeline"

pipe = pickle.load(open(wd, 'rb'))

#%% apply the prediction function

def get_pred(df, column):
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
        24: 'ancient history',
        25: 'poetry',
        26: 'sports'
        }

    X = df[column]
    pred = pipe.transform(X)
    X_trans = np.argmax(pred, axis=1)
    df['topic_num'] = X_trans
    df['topic'] = df['topic_num'].map(topic_names)
    return df
    
topics = get_pred(df_tm, 'combined')


#%% update topics

books.update_table_from_df(table= table, columns= ["topic"], df= topics)

