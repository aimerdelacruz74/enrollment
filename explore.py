import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import io
# %matplotlib inline
import seaborn as sns; sns.set()
from sklearn import preprocessing
fig = plt.rc("font", size = 14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes = True)
import pickle
from sklearn.preprocessing import LabelEncoder
from numpy.core.numeric import True_
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from PIL import Image
import plotly.express as px
import io
from sklearn.model_selection import train_test_split



def show_explore():

    streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css?family=Poppins');

			html, body, [class*="css"]  {
			font-family: 'Poppins', sans-serif;
			}
			</style>
			"""
            
    st.write("""
    # DATA VISUALIZATION
    In this page, Data visualization will be performed to analyze the dataset.
    """)
    
    image = Image.open('soc.jpg')
    st.image(image, use_column_width=True)
    
    st.write("""
    Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, 
    data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.
    """) 

    
    st.markdown(streamlit_style, unsafe_allow_html=True)
    # Sidebar - Collects user input features into dataframe
   
    uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR CSV FILE", type=["csv"])
    st.sidebar.markdown("""
    [SAMPLE CSV FILE](https://drive.google.com/drive/folders/1_z-wXEETysPyu622ESAIA5tpkv_HLfp0?usp=share_link)
    """)
    
   

    

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
        return href
        

    def build_model(df):
    
        st.subheader("DATASET INFORMATION")
        st.write("This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.")
        st.write("")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
       
        st.write("")
        st.subheader("DATAFRAME.DESCRIBE")
        st.write("Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.")
        st.write("")
        des = df.describe(include='all').fillna("").astype("str")
        st.write(des)
        
        

        
        fig = plt.figure()
        df['enrolled'].value_counts().plot(kind="bar")
        st.subheader("BAR GRAPH DISPLAYING THE STATUS OF THE APPLICANTS")
        st.markdown("")
        st.write("This bar graph displays the number of applicants who pursue/did not pursue their enrollment.")
        st.pyplot(fig)
        
        fig1 = plt.figure()
        st.subheader("BAR GRAPH DISPLAYING THE PROVINCE OF THE APPLICANTS")
        st.markdown("")
        st.write("This bar graph displays the information of the applicant's province.")
        df['province'].value_counts().plot(kind="bar")
        st.pyplot(fig1)
        
        st.subheader("BAR GRAPH DISPLAYING THE FINANCIAL SOURCE OF THE APPLICANTS")
        st.markdown("")
        st.write("This bar graph displays the information of the applicant's financial source.")
        fig2 = plt.figure()
        df['finance_src'].value_counts().plot(kind="bar")
        st.pyplot(fig2)
        
        st.subheader("BAR GRAPH DISPLAYING THE TYPE OF CELLPHONE USED BY THE APPLICANTS")
        st.markdown("") 
        st.write("This bar graph displays the information of the applicant's cellphone used.")        
        fig3 = plt.figure()
        df['x2'].value_counts().plot(kind="bar")
        st.pyplot(fig3)
        
        st.subheader("BAR GRAPH DISPLAYING THE HIGHEST EDUCATIONAL ATTAINMENT OF THE APPLICANT'S FATHER")
        st.markdown("")  
        st.write("This bar graph displays the information of the highest educational attainment of the applicant's father.")    
        fig4 = plt.figure()
        df['x3'].value_counts().plot(kind="bar")
        st.pyplot(fig4)

        st.subheader("BAR GRAPH DISPLAYING THE HIGHEST EDUCATIONAL ATTAINMENT OF THE APPLICANT'S MOTHER")
        st.markdown("")
        st.write("This bar graph displays the information of the highest educational attainment of the applicant's mother.")            
        fig5 = plt.figure()
        df['x4'].value_counts().plot(kind="bar")
        st.pyplot(fig5)

        st.subheader("BAR GRAPH DISPLAYING THE MAJOR SOURCE INCOME OF THE APPLICANT'S FATHER")
        st.markdown("")  
        st.write("This bar graph displays the information of the major source of income of the applicant's father.")
        fig6 = plt.figure()
        df['x5'].value_counts().plot(kind="bar")
        st.pyplot(fig6)

        st.subheader("BAR GRAPH DISPLAYING THE MAJOR SOURCE INCOME OF THE APPLICANT'S MOTHER")
        st.markdown("")  
        st.write("This bar graph displays the information of the major source of income of the applicant's mother.")
        fig7 = plt.figure()
        df['x6'].value_counts().plot(kind="bar")
        st.pyplot(fig7)

        st.subheader("BAR GRAPH DISPLAYING THE CLOSEST WORKING STATUS OF THE APPLICANT'S MOTHER")
        st.markdown("")
        st.write("This bar graph displays the information of the applicant's mother working status.")        
        fig8 = plt.figure()
        df['x8'].value_counts().plot(kind="bar")
        st.pyplot(fig8)

        st.subheader("BAR GRAPH DISPLAYING THE OCCUPATION OF THE APPLICANT'S FATHER")
        st.markdown("")
        st.write("This bar graph displays the information of the applicant's father occupation.")
        fig9 = plt.figure()
        df['x9'].value_counts().plot(kind="bar")
        st.pyplot(fig9)

        st.subheader("BAR GRAPH DISPLAYING THE INCOME THAT BEST ESTIMATES THE COMBINED MONTHLY INCOME THE APPLICANT'S FAMILY. (PARENTS, WORKING BROTHERS/SISTERS)")
        st.markdown("")
        st.write("This bar graph displays the information of the applicant's family income per month.")        
        fig10 = plt.figure()
        df['x11'].value_counts().plot(kind="bar")
        st.pyplot(fig10)
        
        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION CLEARLY IDENTIFIED STUDY HABITS/ROUTINES. (5 BEING VERY HIGH)")
        st.markdown("")    
        fig11 = plt.figure()
        df['x26'].value_counts().plot(kind="bar")
        st.pyplot(fig11)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION I LIKE TO WORK IN UNSTRUCUTRED SITUATIONS WHERE I CAN USE MY CREATIVITY)")
        st.markdown("")
        fig12 = plt.figure()
        df['x45'].value_counts().plot(kind="bar")
        st.pyplot(fig12)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION I AM CONCERNED WITH PEOPLE AND THEIR WELFARE. (4 BEING STRONGLY AGREE)")
        st.markdown("")       
        fig13 = plt.figure()
        df['x46'].value_counts().plot(kind="bar")
        st.pyplot(fig13)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION I GIVE MORE FOCUS ON THE POSITIVE SIDE. (4 BEING STRONGLY AGREE)")
        st.markdown("")       
        fig14 = plt.figure()
        df['x67'].value_counts().plot(kind="bar")
        st.pyplot(fig14)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION OPERATING ELECTRONICS DEVICES AND EQUIPMENTS IS MY INTEREST")
        st.markdown("")       
        fig15 = plt.figure()
        df['q1'].value_counts().plot(kind="bar")
        st.pyplot(fig15)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION I AM ABLE TO FACE UP DIFFERENT PROBLEMS. (4 BEING STRONGLY AGREE)")
        st.markdown("")       
        fig16 = plt.figure()
        df['q2'].value_counts().plot(kind="bar")
        st.pyplot(fig16)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION WHEN I DO SOMETHING, I USUALLY WANT TO DO IT ON MY OWN. (4 BEING STRONGLY AGREE)")
        st.markdown("")              
        fig17 = plt.figure()
        df['q3'].value_counts().plot(kind="bar")
        st.pyplot(fig17)

        st.subheader("BAR GRAPH DISPLAYING THE RESULTS OF THE QUESTION I USUALLY USE AVAILABLE RESOURCES TO SOLVE MY PROBLEMS. (4 BEING STRONGLY AGREE)")
        st.markdown("")          
        fig18 = plt.figure()
        df['q5'].value_counts().plot(kind="bar")
        st.pyplot(fig18)

        data = {"lat":[17.5951122,15.1449853,11.8166109,13.1390621,15.1449853,11.380579,16.0773636,14.6416842,13.7564651,16.5577257,
            9.8499911,8.0515054,14.7942735,18.2489629,14.1390265,13.5250197,9.2422662,13.7088684,14.2456329,10.3156992,15.4827722,17.3512542,
            7.1083349,7.5617699,6.7662687,7.512515,7.3171585,10.5928661,16.8330792,18.1647281,17.2278664,10.7201501,12.879721,16.9753758,
            17.4740422,14.1406629,7.8721811,16.6158906,10.8624536,6.9422581,13.4767171,12.3574346,8.5045558,17.0663429,14.6090537,10.247656,
            9.6282083,15.578375,16.3301107,13.1024111,14.8386303,12.879721,9.8349493,15.079409,15.8949055,13.9346903,16.2700424,14.6037446,
            12.5778554,12.2445533,12.9927095,6.5069401,9.514828,8.5404906,15.4754786,15.5081766,6.9214424],
            "lon":[120.7982528,120.5887029,122.0941541,123.7437995,120.5887029,122.0635005,121.7692535,120.4818446,121.0583076,120.8039474,
            124.1435427,124.9229946,120.8799008,121.8787833,122.7633036,123.3486147,124.7351486,124.2421597,120.8785658,123.8854366,
            120.7120023,121.1718851,125.0388164,125.6532848,125.3284269,126.1762615,126.5419887,122.6325081,121.1710389,120.7115592,
            120.5739579,122.5621063,121.774017,121.8107079,121.3541631,121.4691774,123.8857747,120.3209373,124.8811195,124.4198243,
            121.9032192,123.5504076,124.6219592,121.03351,121.0222565,122.9888319,122.9888319,121.1112615,121.1710389,120.7651284,120.2842023,
            121.774017,118.7383615,120.6199895,120.2863183,121.947311,121.5370003,121.3084088,122.269129,125.0388164,124.0147464,124.4198243,
            125.6969984,126.1144758,120.5963492,119.9697808,122.0790267],
            "City": ["ABRA","AGUSAN DEL NORTE","AKLAN","ALBAY","ANGELES","ANTIQUE","AURORA","BATAAN","BATANGAS","BENGUET","BOHOL","BUKIDNON",
            "BULACAN","CAGAYAN","CAMARINES NORTE","CAMARINES SUR","CAMIGUIN","CATANDUANES","CAVITE","CEBU","CENTRAL LUZON","CORDILLERA","COTABATO",
            "DAVAO DEL NORTE","DAVAO DEL SUR","DAVAO DE ORO","DAVAO ORIENTAL","GUIMARAS","IFUGAO","ILOCOS NORTE","ILOCOS SUR","ILOILO","INTERNATIONAL",
            "ISABELA","KALINGA","LAGUNA","LANAO DEL NORTE","LA UNION","LEYTE","MAGUINDANAO","MARINDUQUE","MASBATE","MISAMIS ORIENTAL",
            "MOUNTAIN PROVINCE","NCR","NEGROS OCCIDENTAL","NEGROS ORIENTAL","NUEVA ECIJA","NUEVA VIZCAYA","OCCIDENTAL MINDORO","OLONGAPO","OTHERS",
            "PALAWAN","PAMPANGA","PANGASINAN","QUEZON PROVINCE","QUIRINO","RIZAL","ROMBLON","SAMAR","SORSOGON","SULTAN KUDARAT","SURIGAO DEL NORTE",
            "SURIGAO DEL SUR","TARLAC","ZAMBALES","ZAMBOANGA"]}
         
        loc = pd.DataFrame(data)
        loc
        st.map(data=loc)
        
        labels = ['ABRA','AGUSAN-DEL-NORTE','AKLAN','ALBAY','ANGELES','ANTIQUE','AURORA','BATAAN','BATANGAS','BENGUET','BOHOL',
                        'BUKIDNON','BULACAN','CAGAYAN','CAMARINES-NORTE','CAMARINES-SUR','CAMIGUIN','CATANDUANES','CAVITE','CEBU','CENTRAL-LUZON',
                        'CORDILLERA','COTABATO','DAVAO-DEL-NORTE','DAVAO-DEL-SUR','DAVAO-DE-ORO','DAVAO-ORIENTAL','GUIMARAS','IFUGAO','ILOCOS-NORTE',
                        'ILOCOS-SUR','ILOILO','INTERNATIONAL','ISABELA','KALINGA','LAGUNA','LANAO-DEL-NORTE','LA-UNION','LEYTE','MAGUINDANAO',
                        'MARINDUQUE','MASBATE','MISAMIS-ORIENTAL','MOUNTAIN-PROVINCE','NCR','NEGROS-OCCIDENTAL','NEGROS-ORIENTAL','NUEVA-ECIJA',
                        'NUEVA-VIZCAYA','OCCIDENTAL-MINDORO','OLONGAPO','OTHERS','PALAWAN','PAMPANGA','PANGASINAN','QUEZON-PROVINCE','QUIRINO',
                        'RIZAL','ROMBLON','SAMAR','SORSOGON','SULTAN-KUDARAT','SURIGAO-DEL-NORTE','SURIGAO-DEL-SUR','TARLAC','ZAMBALES','ZAMBOANGA']
                        
        sizes =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
                    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
        
        
        
        
        columnsToEncode = list(df.select_dtypes(include=['category','object']))

        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)


            
    


  
    #---------------------------------#
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
        st.write(df)
        build_model(df)
    else:
        st.info('AWAITING FOR THE CSV FILE TO BE UPLOADED.')
        if st.button('PRESS TO USE EXAMPLE DATASET'):       
            df = pd.read_csv('predictnew2.csv', encoding = 'ISO-8859-1')
            

            st.markdown('THE DATASET FOR S.Y. 2020-2022 IS USED AS THE EXAMPLE.')
            st.write(df)
            
            
            build_model(df)
            
            
            
   
