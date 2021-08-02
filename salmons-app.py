import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Trout Health Prediction
""")
#st.sidebar.header('User Input Features')

#st.sidebar.markdown("""
#[Example CSV input file](example.csv)
#""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)   
#else:
   # def user_input_features():
        
       # ALT = st.sidebar.slider('ALT', 0.0,500.0,14.0)
        #ALP = st.sidebar.slider('ALP', 2.0,2205.0,600.0)
        #AST = st.sidebar.slider('AST', -5.0,500.0,100.0)
        #ALB = st.sidebar.slider('ALB', 1.0,34.0,10.0)
        #LDH = st.sidebar.slider('LDH', 1.0,4240.0,100.0)
        #CK = st.sidebar.slider('CK', 4.0,51606.0,100.0)
        #CK_MB = st.sidebar.slider('CK MB', 20.0,377132.0,100.0)
        #ALDO = st.sidebar.slider('ALDO', -5.0,930.0,100.0)
       # CREA = st.sidebar.slider('CREA', 6.0,640.0,200.0)
        #P= st.sidebar.slider('P', 0.2,26.0,12.0)
        #CO2= st.sidebar.slider('CO2', 0.2,26.0,10.0)
        #AMM= st.sidebar.slider('AMM', 85.0,3018.0,100.0)
        #NA_K= st.sidebar.slider('NA/K', 55.0,876.0,100.0)
        #data = {'ALT':ALT,
                #'ALP':ALP,
               # 'AST':AST,
               # 'ALB':ALB,
               # 'LDH':LDH,
                #'CK':CK,
                #'CK':CK_MB,
                #'ALDO':ALDO,
                #'CREA':CREA,
                #'P':P,
                #'CO2':CO2,
                #'AMM':AMM,
                #'NA/K':NA_K}
        #features = pd.DataFrame(data, index=[0])
        #return features
    #input_df = user_input_features()
    #input_df = 0

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
salmons_raw = pd.read_csv('https://raw.githubusercontent.com/Shahadate-Rezvy/penguins-heroku/master/classwise-average-filling_reduced.csv')
salmons = salmons_raw.drop(columns=['JB_category','New_category','Month'], axis=1)

df = pd.concat([input_df,salmons],axis=0)
df.fillna(df.median(), inplace=True)
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)
    
# Reads in saved classification model
load_clf = pickle.load(open('salmons_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
salmon_health = np.array(['HEALTHY','UNHEALTHY'])
st.write(salmon_health[prediction])

#st.subheader('Prediction Probability')
#st.write(prediction_proba)
