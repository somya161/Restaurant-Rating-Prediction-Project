#!/usr/bin/env python
# coding: utf-8
"""
Created on Sat Dec 24 20:27:17 2022

@author: somya
"""
# ## Deployment




import logging as lg
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.title('Zomato Restaurant Rating Prediction')

select = st.selectbox(label='Select one of the following',options=['About','Predictions'])

if select=='About':
    submit = st.button(label='Submit')
    if submit:
        st.header('About Project')
        st.write(
            "Problem Statement: The main goal of this project is to perform extensive Exploratory Data Analysis(EDA) on the Zomato Dataset and build an appropriate Machine Learning Model that will help various Zomato Restaurants to predict their respective Ratings based on certain features.")
            
class preprocessing:
    def __init__(self):
        self.logger = lg

    def preprocess(self,df):
        try:
            lg.info('preprocessing')
            df.drop(['url', 'address', 'name', 'phone', 'reviews_list', 'menu_item', 'listed_in(city)', 'dish_liked'],
                axis=1, inplace=True)
            df['rate'].replace('NEW', np.NaN, inplace=True)
            df['rate'].replace('-', np.NaN, inplace=True)
            df['rate'] = df['rate'].astype(str)
            df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
            df['rate'] = df['rate'].apply(lambda x: float(x))
            df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)
            df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: x.replace(',', '.'))
            df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)
            df.rename(columns={'approx_cost(for two people)': 'approx_cost'}, inplace=True)
            df['rate'].fillna(df['rate'].mean(), inplace=True)
            df['location'].fillna(df['rate'].mode()[0], inplace=True)
            map_ = {'Yes': 1, 'No': 0}
            df['online_order'] = df['online_order'].map(map_)
            df['book_table'] = df['book_table'].map(map_)
            type_ = {'Buffet': 1, 'Cafes': 2, 'Delivery': 3, 'Desserts': 4, 'Dine-out': 5, 'Drinks & nightlife': 6,
                 'Pubs and bars': 7}
            df['listed_in(type)'] = df['listed_in(type)'].map(type_)
            df.rename(columns={'listed_in(type)':'listed_intype'},inplace=True)
            df.dropna(inplace=True)
            le = LabelEncoder()
            df.location = le.fit_transform(df.location)
            df.rest_type = le.fit_transform(df.rest_type)
            df.cuisines = le.fit_transform(df.cuisines)
            lg.info('train preprocessing successful')
            return df
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

obj1 = preprocessing()
zom = pd.read_csv('zomato.csv')

df = obj1.preprocess(zom)

class split:
    def __init__(self):
        self.logger = lg
    def train_split(self,df):
        try:
            lg.info('train split')
            x = df[['online_order','book_table','votes','location','rest_type','cuisines','approx_cost','listed_intype']]
            y = df['rate']
            lg.info('split successful')
            print(type(x))
            return [x,y]
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

obj2 = split()
x1 = obj2.train_split(df)
#y2 = obj3.train_split(df)
x=x1[0]
y=x1[1]
class prediction:
    def __init__(self):
        self.logger = lg
    def predict(self,x,y,test):
        try:
            lg.info('Random Forest algorithm')
            rf = RandomForestRegressor()
            rf.fit(x,y)
            y_pred = rf.predict(test)
            lg.info('training successful')
            return y_pred
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

class test_preprocess:
    def __init__(self):
        self.logger = lg
    def test_preprocess1(self,test_df):
        try:
            lg.info('test preprocessing')
            #test_df['approx_cost'] = test_df['approx_cost'].astype(str)
            #test_df['approx_cost'] = test_df['approx_cost'].apply(lambda x: x.replace(',', '.'))
            #test_df['approx_cost'] = test_df['approx_cost'].astype(float)
            map1 = {'Yes': 1, 'No': 0}
            test_df.online_order = test_df.online_order.map(map1)
            test_df.book_table = test_df.book_table.map(map1)
            type_ = {'Buffet': 1, 'Cafes': 2, 'Delivery': 3, 'Desserts': 4, 'Dine-out': 5, 'Drinks & nightlife': 6,
                 'Pubs and bars': 7}
            test_df.listed_intype = test_df.listed_intype.map(type_)
            le = LabelEncoder()
            test_df.location = le.fit_transform(test_df.location)
            test_df.rest_type = le.fit_transform(test_df.rest_type)
            test_df.cuisines = le.fit_transform(test_df.cuisines)
            lg.info('test preprocessing successful')
            return test_df
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

if select=='Predictions':
    zom = pd.read_csv('zomato.csv')
    online_order = st.selectbox(label='online_order',options=['Yes','No'])
    book_table = st.selectbox(label='book_table',options=['Yes','No'])
    votes = st.number_input(label='votes')
    location = st.selectbox(label='location',options=zom.location.unique().tolist())
    rest_type = st.selectbox(label='rest_type',options=zom.rest_type.unique().tolist())
    cuisines = st.selectbox(label='cuisines',options=zom.cuisines.unique().tolist())
    approx_cost = st.number_input(label='approximate cost for two')
    listed_intype = st.selectbox(label='listed in type',options=zom['listed_in(type)'].unique().tolist())

    submit = st.button(label='Get predictions')
    if submit:
        test_df1 = pd.DataFrame([[online_order,book_table,votes,location,rest_type,cuisines,approx_cost,listed_intype]],columns=['online_order','book_table','votes','location','rest_type','cuisines','approx_cost','listed_intype'])
        #st.dataframe(test_df1)
        obj3 = test_preprocess()
        test_df = obj3.test_preprocess1(test_df1)
        obj4 = prediction()
        pred = obj4.predict(x,y,test_df)
        st.write('The predicted rating is {}'.format(pred))

