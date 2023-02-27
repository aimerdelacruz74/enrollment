import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import io
# %matplotlib inline
import seaborn as sns; sns.set()
from sklearn import preprocessing
fig = plt.rc("font", size = 14)
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes = True)
import pickle
from numpy.core.numeric import True_
from sklearn import metrics




def show_ranfor():

    streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css?family=Poppins');

			html, body, [class*="css"]  {
			font-family: 'Poppins', sans-serif;
			}
			</style>
			"""
    st.markdown(streamlit_style, unsafe_allow_html=True)
    
    st.write("""
    # RANDOM FOREST
    In this page, the RANDOM FOREST  function is used in this app.
    """)
    
    image = Image.open('RANFOR.PNG')
    st.image(image, use_column_width=True)
    
    st.write("""
    Random Forest  is a supervised learning algorithm that uses ensemble learning method for regression. 
    Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms 
    to make a more accurate prediction than a single model.
    """)
    
    st.write("""
    Random forests present estimates for variable importance, i.e., neural nets. 
    They also offer a superior method for working with missing data. Missing values are substituted by the 
    variable appearing the most in a particular node. Among all the available classification methods, 
    random forests provide the highest accuracy.
    """)

    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR CSV FILE", type=["csv"])
    st.sidebar.markdown("""
    [SAMPLE CSV FILE](https://drive.google.com/drive/folders/1_z-wXEETysPyu622ESAIA5tpkv_HLfp0?usp=share_link)
    """)

    # Sidebar - Specify parameter settings
    st.sidebar.subheader("HYPERPARAMETERS")        
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
    #---------------------------------#
    # Main panel




    #---------------------------------#
    # Model building

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
        return href
        



    def build_model(df):
        X = df.iloc[:,:-1] # Using all column except for the last column as X
        Y = df.iloc[:,-1] # Selecting the last column as Y
        
        le_province = LabelEncoder()
        df ['province'] = le_province.fit_transform(df['province'])
        df["province"].unique()

        le_finance_src = LabelEncoder()
        df ['finance_src'] = le_finance_src.fit_transform(df['finance_src'])
        df["finance_src"].unique()

        le_x2 = LabelEncoder()
        df ['x2'] = le_x2.fit_transform(df['x2'])
        df["x2"].unique()

        le_x3 = LabelEncoder()
        df ['x3'] = le_x3.fit_transform(df['x3'])
        df["x3"].unique()

        le_x4 = LabelEncoder()
        df ['x4'] = le_x4.fit_transform(df['x4'])
        df["x4"].unique()

        le_x5 = LabelEncoder()
        df ['x5'] = le_x5.fit_transform(df['x5'])
        df["x5"].unique()

        le_x6 = LabelEncoder()
        df ['x6'] = le_x6.fit_transform(df['x6'])
        df["x6"].unique()

        le_x8 = LabelEncoder()
        df ['x8'] = le_x8.fit_transform(df['x8'])
        df["x8"].unique()

        le_x9 = LabelEncoder()
        df ['x9'] = le_x9.fit_transform(df['x9'])
        df["x9"].unique()

        le_x11 = LabelEncoder()
        df ['x11'] = le_x11.fit_transform(df['x11'])
        df["x11"].unique()


        le_x45 = LabelEncoder()
        df ['x45'] = le_x45.fit_transform(df['x45'])
        df["x45"].unique()

        le_x46 = LabelEncoder()
        df ['x46'] = le_x46.fit_transform(df['x46'])
        df["x46"].unique()

        
        X = df.drop ('enrolled', axis = 1)
        y = df['enrolled'] 

        st.markdown('A MODEL IS BEING BUILT TO PREDICT THE FOLLOWING **Y** VARIABLE:')
        st.info(Y.name)
        
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
       
        trainingSet, testSet = train_test_split(df, test_size=0.2)

        train_df = trainingSet
        test_df = testSet

        X_train = train_df[['province','finance_src','x2','x3','x4','x5','x6','x8','x9','x11','x26','x45','x46','x67','q1','q2','q3','q5']] 	

        y_train = train_df['enrolled']

        X_test = test_df[['province','finance_src','x2','x3','x4','x5','x6','x8','x9','x11','x26','x45','x46','x67','q1','q2','q3','q5']] 	

        y_test = test_df['enrolled']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        def plot_metrics(metrics_list):
        
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=   class_names)
            st.pyplot(fig)
            

            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot(fig)
            
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot(fig)
            
        class_names = ["Actual", "Predicted"]

        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators = n_estimators, criterion='entropy',random_state=0,max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        accuracy = model.score(X_test.values, y_test.values)
        
        st.write("Accuracy: ", metrics.accuracy_score(y_test.values, y_pred).round(2))
        st.write("Precision: ", precision_score(y_test.values, y_pred, labels = class_names).round(2))
        st.write("Recall: ", recall_score(y_test.values, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)


    #---------------------------------#
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
        st.write(df)
        build_model(df)
    else:
        st.info('AWAITING FOR THE CSV FILE TO BE UPLOADED.')
        if st.button('PRESS TO USE EXAMPLE DATASET'):    
            df = pd.read_csv('predictnew.csv', encoding = 'ISO-8859-1')
            X = df.drop ('enrolled', axis = 1)
            Y = df['enrolled']
            df = pd.concat( [X,Y], axis=1 )
            


            st.markdown('THE DATASET FOR S.Y. 2020-2022 IS USED AS THE EXAMPLE.')
            st.write(df.head(5))

            build_model(df)

