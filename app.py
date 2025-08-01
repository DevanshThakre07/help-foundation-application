import streamlit as st
import numpy as np
import pandas as pd
import joblib 
# First load the instances that wer created
with open('scaler.joblib','rb') as file:
    scale = joblib.load(file)
with open("pca.joblib","rb") as file:
    pca = joblib.load(file)
with open("final_model.joblib","rb") as file:
    model = joblib.load(file)
def prediction(input_list):
    scale_input = scale.transform([input_list])
    pca_input = pca.transform(scale_input)
    output = model.predict(pca_input)[0]
    if output == 0:
        return 'Underdeveloped'
    elif output == 1:
        return 'Developed'
    else:
        return 'Developing'
def main():
    st.title('Help NGO Foundation')
    st.subheader('This application will give the status of a countary based on socio-economis and healthcare')
    gdp = st.text_input('Enter the GDP per population of a country')
    inc = st.text_input('Enter the  per capita income of a country')
    imp = st.text_input('Enter the  imports in terms of GDP')
    exp = st.text_input('Enter the  exports in terms of GDP')
    inf = st.text_input('Enter the  inflation in the country in percentage')
    hel = st.text_input('Enter the  expenditure on health in terms % of GDP')
    ch_m = st.text_input('Enter the  no of deaths per 1000 births for <5 yrs')
    fer = st.text_input('Enter the avg children born to a women in a country')
    lf = st.text_input('Enter the avg life expectency in a country')
    in_data = [ch_m,exp,hel,imp,inc,inf,lf,fer,gdp]

    if st.button('Predict'):
        response = prediction(in_data)
        st.success(response)
if __name__=='__main__':
    main()
