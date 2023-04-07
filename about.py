import streamlit as st
from PIL import Image


def show_about():


    streamlit_style = """
        <style>
        @import url('https://fonts.googleapis.com/css?family=Poppins');

        html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        }
        </style>
        """
    st.markdown(streamlit_style, unsafe_allow_html=True)

    st.title("Predictive Modeling of Enrollment Likelihood in Holy Angel University Using Machine Learning and Personality Assessment: A Comparative Analysis of Logistic Regression, Random Forest, and Support Vector Machine")
    st.write("")
    image = Image.open('gs.jpg')
    st.image(image, use_column_width=True)
    st.write("")
    st.write("This website is a machine learning based application that determines the probability of likelihood of the applicants to continue their admission to the university based on their personality assessment")
    st.write("")
