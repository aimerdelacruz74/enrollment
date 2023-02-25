import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import LabelEncoder 
#from predict2 import predict  

def load_model():
    with open ('saved_steps3.pkl','rb') as file:
        data = pickle.load(file)
    return data
	


def show_predict_page():

    streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css?family=Poppins');

			html, body, [class*="css"]  {
			font-family: 'Poppins', sans-serif;
			}
			</style>
			"""
    st.markdown(streamlit_style, unsafe_allow_html=True)
    #st.title("ENROLLMENT PREDICTION")
    st.write("""### INPUT INFORMATION""")
    st.markdown("")
    # Sidebar - Collects user input features into dataframe

    uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR CSV FILE", type=["csv"])
    st.sidebar.markdown("""
    [SAMPLE CSV FILE](https://drive.google.com/drive/folders/1_z-wXEETysPyu622ESAIA5tpkv_HLfp0?usp=share_link)
    """)
    
    st.sidebar.markdown("""
    THE RANDOM FOREST REGRESSOR WAS USED ON THIS PREDICTION.
    """)
    st.sidebar.markdown("""
    TO KNOW MORE ABOUT THIS MODEL, YOU MAY PROCEEED TO THE EXPLORE PAGE.
    """)
    
    
    province = ('ABRA','AGUSAN-DEL-NORTE','AKLAN','ALBAY','ANGELES','ANTIQUE','AURORA','BATAAN','BATANGAS','BENGUET','BOHOL','BUKIDNON','BULACAN','CAGAYAN','CAMARINES-NORTE','CAMARINES-SUR','CAMIGUIN','CATANDUANES','CAVITE','CEBU','CENTRAL-LUZON','CORDILLERA','COTABATO','DAVAO-DEL-NORTE','DAVAO-DEL-SUR','DAVAO-DE-ORO','DAVAO-ORIENTAL','GUIMARAS','IFUGAO','ILOCOS-NORTE','ILOCOS-SUR','ILOILO','INTERNATIONAL','ISABELA','KALINGA','LAGUNA','LANAO-DEL-NORTE','LA-UNION','LEYTE','MAGUINDANAO','MARINDUQUE','MASBATE','MISAMIS-ORIENTAL','MOUNTAIN-PROVINCE','NCR','NEGROS-OCCIDENTAL','NEGROS-ORIENTAL','NUEVA-ECIJA','NUEVA-VIZCAYA','OCCIDENTAL-MINDORO','OLONGAPO','OTHERS','PALAWAN','PAMPANGA','PANGASINAN','QUEZON-PROVINCE','QUIRINO','RIZAL','ROMBLON','SAMAR','SORSOGON','SULTAN-KUDARAT','SURIGAO-DEL-NORTE','SURIGAO-DEL-SUR','TARLAC','ZAMBALES','ZAMBOANGA')
    finance_src = ('PARENTS','GUARDIAN','BROTHER','SISTER','SELF-SUPPORTING','NOT-APPLICABLE')
    x2 = ('ANALOG','ANDROID','IOS','ANDROID-AND-IOS')
    x3 = ('ELEM','HS','COLLEGE','POSTGRAD','NOT-APPLICABLE')
    x4 = ('ELEM','HS','COLLEGE','POSTGRAD','NOT-APPLICABLE')
    x5 = ('SALARY','BUSINESS','OFW','PROFESSIONAL','FARM','NOT-APPLICABLE')
    x6 = ('SALARY','BUSINESS','OFW','PROFESSIONAL','FARM','NOT-APPLICABLE')
    x8 = ('FULLTIME','PARTTIME','CASUAL','RETIRED','NONE','NOT-APPLICABLE')
    x9 = ('ACCOUNT-OFFICE','ARCH-DESIGN-GRAPHIC','DECEASED','DRIVER','ENGR-SKILLED-TECH','HRM-BUSINESS','LABORER','LAW-TEACHING-SOCIAL','MEDICAL','NONE','OFW','RETIRED','SELF-EMPLOYED')
    x11 =('BELOW-15K','16K-31K','32K-78K','79K-117K','118K-157K','158K-ABOVE')
    x26 = ('1','2','3','4','5')
    x45 = ('YES','NO')
    x46 = ('YES','NO','NOT-APPLICABLE')
    x67 = ('1','2','3','4')
    q1 = ('1','2','3','4')
    q2 = ('1','2','3','4')
    q3 =('1','2','3','4')
    q5 = ('1','2','3','4')

    
        #---------------------------------#
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
        #predict()
        le_province = LabelEncoder()
        df ['province'] = le_province.fit_transform(df['province'])
        df['province'].unique()

        le_finance_src = LabelEncoder()
        df ['finance_src'] = le_finance_src.fit_transform(df['finance_src'])
        df['finance_src'].unique()

        le_x2 = LabelEncoder()
        df ['x2'] = le_x2.fit_transform(df['x2'])
        df['x2'].unique()

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
        Y = df['enrolled']
        X = df.drop ('enrolled', axis = 1)



        trainingSet, testSet = train_test_split(df, test_size=0.2)

        train_df = trainingSet
        test_df = testSet

        X_train = train_df[['province','finance_src','x2','x3','x4','x5','x6','x8','x9','x11','x26','x45','x46','x67','q1','q2','q3','q5']] 	
        y_train = train_df['enrolled']
        X_test = test_df[['province','finance_src','x2','x3','x4','x5','x6','x8','x9','x11','x26','x45','x46','x67','q1','q2','q3','q5']] 	
        y_test = test_df['enrolled']



        
    else:
        st.info('AWAITING FOR CSV FILE TO BE UPLOADED')
        st.markdown("")
        if st.button('PRESS TO USE EXAMPLE DATASET'):    
            df = pd.read_csv('predictnew.csv', encoding = 'ISO-8859-1')
  
            st.markdown('THE DATASET FOR S.Y. 2020-2022 IS USED AS THE EXAMPLE.')
            
            
            load_model()
            
            
            
            


    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)

    province = st.selectbox("PROVINCE",province)
    st.markdown("")
    finance_src = st.selectbox("FINANCIAL SOURCE",finance_src)
    st.markdown("")
    x2 = st.selectbox("WHAT TYPE OF CELLPHONE DO YOU USE?",x2)
    st.markdown("")
    x3 = st.selectbox("WHICH IS THE HIGHEST EDUCATIONAL ATTAINMENT OF YOUR FATHER?",x3)
    st.markdown("")
    x4 = st.selectbox("WHICH IS THE HIGHEST EDUCATIONAL ATTAINMENT OF YOUR MOTHER?",x4)
    st.markdown("")
    x5 = st.selectbox("WHAT IS THE MAJOR SOURCE OF INCOME OF YOUR FATHER?",x5)
    st.markdown("")
    x6 = st.selectbox("WHAT IS THE MAJOR SOURCE OF INCOME OF YOUR MOTHER",x6)
    st.markdown("")
    x8 = st.selectbox("WHICH CLOSELY DESCRIBES YOUR MOTHER'S WORKING STATUS?",x8)
    st.markdown("")
    x9 = st.selectbox("WHICH CLOSELY DESCRIBES YOUR FATHER'S OCCUPATION?",x9)
    st.markdown("")
    x11 = st.selectbox("PLEASE CHOOSE THE INCOME THAT BEST ESTIMATES THE COMBINED MONTHLY INCOME OF YOUR FAMILY. (PARENTS, WORKING BROTHERS/SISTERS)",x11)
    st.markdown("")
    x26 = st.radio("CLEARLY IDENTIFIED STUDY HABITS/ROUTINES. (5 BEING VERY HIGH)",x26)
    st.markdown("")
    x45 = st.radio("I LIKE TO WORK IN UNSTRUCUTRED SITUATIONS WHERE I CAN USE MY CREATIVITY.",x45)
    st.markdown("")
    x46 = st.radio("I AM CONCERNED WITH PEOPLE AND THEIR WELFARE. (4 BEING STRONGLY AGREE)",x46)
    st.markdown("")
    x67 = st.radio("I GIVE MORE FOCUS ON THE POSITIVE SIDE. (4 BEING STRONGLY AGREE)",x67)
    st.markdown("")
    q1 = st.radio("OPERATING ELECTRONICS DEVICES AND EQUIPMENTS IS MY INTEREST. (4 BEING STRONGLY AGREE)",q1)
    st.markdown("")
    q2 = st.radio("I AM ABLE TO FACE UP DIFFERENT PROBLEMS. (4 BEING STRONGLY AGREE)",q2)
    st.markdown("")
    q3 = st.radio("WHEN I DO SOMETHING, I USUALLY WANT TO DO IT ON MY OWN. (4 BEING STRONGLY AGREE)",q3)
    st.markdown("")
    q5 = st.radio("I USUALLY USE AVAILABLE RESOURCES TO SOLVE MY PROBLEMS. (4 BEING STRONGLY AGREE)",q5)
    

    
    
  
    ok = st.button("PREDICT")
    if ok:
    
        data = load_model()
	
        clf= data ["model"]
        le_province = data["le_province"]
        le_finance_src = data["le_finance_src"]
        le_x2 = data["le_x2"]
        le_x3 = data ["le_x3"]
        le_x4 = data ["le_x4"]
        le_x5 = data ["le_x5"]
        le_x6 = data ["le_x6"]
        le_x8 = data ["le_x8"]
        le_x9 = data ["le_x9"]
        le_x11 = data["le_x11"]
        le_x45 = data["le_x45"]
        le_x46 = data["le_x46"]
        
        X = np.array([[province,finance_src,x2,x3,x4,x5,x6,x8,x9,x11,x26,x45,x46,x67,q1,q2,q3,q5]])
        X[:, 0] = le_province.transform(X[:,0])
        X[:, 1] = le_finance_src.transform(X[:,1])
        X[:, 2] = le_x2.transform(X[:,2])
        X[:, 3] = le_x3.transform(X[:,3])
        X[:, 4] = le_x4.transform(X[:,4])
        X[:, 5] = le_x5.transform(X[:,5])
        X[:, 6] = le_x6.transform(X[:,6])
        X[:, 7] = le_x8.transform(X[:,7])
        X[:, 8] = le_x9.transform(X[:,8])
        X[:, 9] = le_x11.transform(X[:,9])
        X[:, 11] = le_x45.transform(X[:,11])
        X[:, 12] = le_x46.transform(X[:,12])

        X = X.astype(float)

        enrolled = clf.predict(X)
        st.subheader(f"PROBABILITY TO ENROLL: {enrolled[0]:.3f}%")   