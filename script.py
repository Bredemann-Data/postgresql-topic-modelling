#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:48:10 2023

@author: basti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:19:10 2023

@author: basti
"""

import psycopg2 as pg
import pandas as pd
import numpy as np

#import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline

#%%

class from_sql():
    def __init__(self,database):
        self.database = database
        self.connector = pg.connect(dbname= self.database, user="postgres", password = "Yankees27")
    
    def connect(self):
        return pg.connect(dbname= self.database, user="username", password = "your password")
    
    
    def query_to_df(self, table, columns=None):
        conn = self.connect()
        with conn.cursor() as curs:
            
            curs.execute(f"""select * FROM "{table}";""")
            
            rows = curs.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=columns)

    def write_to_server(self, table, columns, values):
        c_quote = ["\"" + c + "\"" for c in columns]
        # columns in comma list for insert into
        c_string = ", ".join(c_quote)
        # put values into correct quotation marks
        v_quote = ["'" + f"{v}" + "'" for v in values]
        v_string = ", ".join(v_quote)
        conn = self.connect()
        with conn.cursor() as curs:
            
            curs.execute("INSERT INTO " + table + " (" + c_string + ") VALUES (" + v_string + ");")
        conn.commit()
        conn.close()
    
    def write_df_to_server(self, table, columns, df):
        c_quote = ["\"" + c + "\"" for c in columns]
        c_string = ", ".join(c_quote)
        
        conn = self.connect()
        curs = conn.cursor()
        
        for row in df.iterrows():
            v_quote = ["'" + f"{v}" + "'" for v in row[1]]
            v_string = ", ".join(v_quote)
            curs.execute("INSERT INTO " + f'"{table}"' + " (" + c_string + ") VALUES (" + v_string + ");")
            conn.commit()
        
        curs.close()
        conn.close()
        

    def create_table(self, table, column_dtype):
        conn = self.connect()
        with conn.cursor() as curs:
            
            curs.execute(f"""CREATE TABLE "{table}" {column_dtype};""")
        conn.commit()
        conn.close()
            
        
#%% Daten laden

books = from_sql("Book_reviews")

view = "book_title_summary"


df = books.query_to_df(table= view, columns= ["Tite", "ID", "description", "categories"])
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
    #return df

add_lang_column(df=df, spalte="description", path='language')

# check statistics
df.value_counts()

df.to_csv("languages.csv", columns=['ID', 'lang'], index=False)

#%% Filter for language

df_en = df.copy()

df_en['lang'] = pickle.load(open('language.p', 'rb'))

df_en = df_en[df_en['lang'] == "en"]

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

#%% topic modelling with LDA
def topic_modeller(data, n_topics, file_name):
    vec = CountVectorizer(stop_words=stops, max_df=0.95, min_df=2)
    LDA = LatentDirichletAllocation(n_components=n_topics, random_state=13, n_jobs= -1 
                                    )
    pipe = make_pipeline(vec, LDA)
    # train pipeline
    pipe.fit(data)
    # dump pipeline
    #pickle.dump(pipe, open(file_name + ".p", "wb"))
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
pipe = pickle.load(open('pipe.p', 'rb'))
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
    24: 'ancient history',
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

column_dtype = """("ID" integer, topic character varying(25) )"""

#books.create_table("topics_2", column_dtype=column_dtype)

books.write_df_to_server(table= "topics_2", columns= ["ID", "topic"], df = topics[['ID', 'topic_name']])

