# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:17:36 2025

@author: Dell
"""

import streamlit as st
import joblib

vectorizer = joblib.load("vectorization.jb")
model = joblib.load('./GBC.jb')

st.title("Real or Fake news Analysis")
st.write("Enter a news Article below")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        
        
        if prediction[0]==1:
            st.success("The News is Real ")
        else:
            st.error("The News is Fake ")
    else:
        st.warning("Please enter some text to analize. ")