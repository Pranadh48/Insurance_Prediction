import streamlit as st
from src.prediction import Insurance_Prediction

st.title("Insurance Prediction App")

Age = st.number_input("Enter age: ", min_value=0, max_value=120, step=1)
Annual_Income_LPA = st.number_input("Enter Annual Income in LPA: ")
Policy_Term_Years = st.number_input("Enter Policy term in years: ")
Sum_Assured_Lakhs = st.number_input("Enter sum assured in lakhs: ") 


if st.button("Predict"):
    model = Insurance_Prediction()
    result = model.prediction_model(Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs)
    st.success(result)