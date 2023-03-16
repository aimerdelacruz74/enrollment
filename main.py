import streamlit as st
from predict_page import show_predict_page
from ranfor import show_ranfor
from log_reg import show_log_reg
from svm import show_svm
from explore import show_explore
from about import show_about
from PIL import Image
from pathlib import Path
import hydralit_components as hc



st.set_page_config(layout="wide")



def app():

    
             
            
            
    st.markdown("<h1 style='text-align: center;'>HOLY ANGEL UNIVERSITY</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>ENROLLMENT FORECASTING  </h2>", unsafe_allow_html=True)


    image = Image.open('main.png')
    st.image(image, use_column_width=True)
    

    st.sidebar.markdown("# HOME")
    streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css?family=Poppins');

			html, body, [class*="css"]  {
			font-family: 'Poppins', sans-serif;
			}
			</style>
			"""
    st.markdown(streamlit_style, unsafe_allow_html=True)


    st.write("""Holy Angel University has over 16,000 students, making it one of the biggest schools in Central Luzon. 
        Established in 1933 by local philanthropist Juan D. Nepomuceno, it is considered the Philippines' first Catholic 
        school founded by a layman instead of the usual diocese or religious congregation. It is also the first Catholic
        high school in the country that is co-educational, instead of the usual exclusively for boys or exclusively for girls.""")
        
    st.write("""HAU offers cutting-edge academic programs from Basic Education to Graduate School and has a 
        package of scholarships and grants programs for qualified and deserving applicants. In its 9th decade, HAU continues to 
        provide a remarkable campus experience for every Angelite. """)
        
    st.write("""The purpose of this Machine Learning web based application is to predict the likelihood of incoming college freshmen 
        students to enroll at Holy Angel University by basing it on their demographic information and personality exam to 
        determine the possible number of enrollees. """)
        
    

    
 
def predict_page():
    st.markdown("# ENROLLMENT PREDICTION")
    st.sidebar.markdown("# PREDICT️")
    show_predict_page()

def model_fitting():
    st.markdown("# MODEL FITTING")
    st.sidebar.markdown("# MODEL FITTING")
    classifier = st.sidebar.selectbox("CHOOSE CLASSIFIER", ("SUPPORT VECTOR MACHINE (SVM)", "LOGISTIC REGRESSION", "RANDOM FOREST"))
    if classifier == "RANDOM FOREST":
        show_ranfor()     

    if classifier == "LOGISTIC REGRESSION":
       show_log_reg()
       
    if classifier == "SUPPORT VECTOR MACHINE (SVM)":
        show_svm()
        
def explore():
    st.markdown("# EXPLORE")
    st.sidebar.markdown("# EXPLORE️")
    show_explore()

def about():
    st.sidebar.markdown("# ABOUT")
    show_about()

page_names_to_funcs = {
    "HOME": app,
    "PREDICT": predict_page,
    "MODEL FITTING": model_fitting,
    "EXPLORE": explore,
    "ABOUT": about
}

selected_page = st.sidebar.selectbox("SELECT A PAGE", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
