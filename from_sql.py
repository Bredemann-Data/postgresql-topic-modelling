#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:01:48 2023

@author: basti
"""

import psycopg2 as pg
import pandas as pd
import numpy as np

#import spacy
#%% 

class from_sql():
    def __init__(self):
        self.database = input("database: ")
        self.user = input("username: ")
        self.password = input("password: ")
    
    def connect(self):
        return pg.connect(dbname= self.database, user= self.user, password = self.password)

# Method that writes the output of an SQL query into a pd.DataFrame
    def query_to_df(self, table, condition, *columns):
        c_quote = ["\"" + c + "\"" for c in columns]
        c_string = ", ".join(c_quote)
        conn = self.connect()
        with conn.cursor() as curs:
            curs.execute(f"""select {c_string} FROM "{table}" WHERE {condition};""")
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
        

    def create_table(self, table, **column_dtype):
        """
        table: name for table in database
        **column_dtype:
            variable name = name for column in database
            value: string that specifies the datatype for the db column
        """
        string_c_d = ", ".join([c + " " + column_dtype[c] for c in column_dtype])
        conn = self.connect()
        with conn.cursor() as curs:
            
            curs.execute(f"""CREATE TABLE "{table}" ({string_c_d});""")
        conn.commit()
        conn.close()
        return
    
# method inserting the values from a pandas.DataFrame into a table in self.database
    def update_table_from_df(self, table, columns, df):
        
        """
        
        table = table in the database to be updated.
        df = the dataframe that contains the data to be added to database.
        columns = column in the dataframe that contain the values for updating.
        
        This method takes the values specified in a df column and adds them to
        a table in a database. The rows in the postgreSQL table and the pandas dataframe are matched by 
        the ID column (note that this is NOT the index of the dataframe !!).
        The postgreSQL table must contain a column with the same name as the 
        df column that contains the values for upload.
        
        
        """
        
        print ("starting updating process")
        c_quote = ["\"" + c + "\"" for c in columns]
        
        conn = self.connect()
        curs = conn.cursor()
        
        for row in df.iterrows():
            v_quote = ["'" + f"{v}" + "'" for v in row[1][columns]]
            # create zip
            cols_vals = zip(c_quote, v_quote)
            # create list from zip
            cols_vals = [f"{c} = {v}" for c,v in cols_vals]
            # create string from zip
            c_v_str = ", ".join(cols_vals)
            
            curs.execute("UPDATE " + f'"{table}" ' +
                         "SET " + c_v_str + 
                         f""" Where "ID" = {row[1]["ID"]}""")
            conn.commit()
            print("finished row")
        curs.close()
        conn.close()
        return
