#loading required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = "whitegrid")
import pickle
import urllib.request
import importlib
import altair
import time
from PIL import Image

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from log_reg import *


@st.cache
def load_data():
    # assume data file will always be the same per training
    data = pickle.load(open('./df.pkl', 'rb'))
    return data

# callig model saved in Jupyter notebook
pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.sidebar.title("Navigation")
rad = st.sidebar.radio("Go to",["Home","Dataset","Models","Cardiovascular Predictor"])

if rad == "Home":
    st.title("Cardiovascular Disease App")
    st.image('cardio.jpeg', width=None)
    st.write("addd text dfkjdfkdjfdkjfkdjfkdjfdkjf")
    
if rad == "Dataset":
    st.write("addd text dfkjdfkdjfdkjfkdjfkdjfdkjf")
    st.subheader('Dataset')

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)

    st.write(df)

    st.balloons()

 

if rad == "Models":
   st.title('Models')
   st.write("[SVM] (https://nbviewer.jupyter.org/github/curlylady321/Final_Project_CardioVascular-Disease/blob/main/Cardio_data_analysis.ipynb)") 
   st.write("[Random Forrest] (https://nbviewer.jupyter.org/github/jenniferfernandezcadiz/Final_Project/blob/main/Random_Forest_JF.ipynb)")
   st.write("Logistic Regression")

if rad == "Cardiovascular Predictor":
    st.title('Cardiovascular Disease Prediction')
    name = st.text_input("Name:")
    age = st.number_input("Age:")
    gender = st.number_input("Gender:  1 female | 2 male")
    bmi =  st.number_input("BMI | Body mass index (weight in kg/(height in m)^2):")
    st.markdown("[BMI Calculator] (https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/english_bmi_calculator/bmi_calculator.html)")
    ap_hi = st.number_input("Systolic blood pressure:")
    ap_lo = st.number_input("Diastolic blood pressure:")
    cholesterol = st.number_input("Cholesterol: 1 normal | 2 above normal | 3 well above normal")
    gluc = st.number_input("Glucose: 1 normal | 2 above normal | 3 well above normal")
    smoke = st.number_input("Smoking:  0 no | 1 yes")
    alco = st.number_input("Alcohol Intake:  0 no | 1 yes")
    active = st.number_input("Physical activity: 0 no | 1 yes")
    submit = st.button('Predict')

    if submit:
        prediction = classifier.predict([[age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        if prediction == 0:
            st.write(name,"-- Based on your inputs, you are NOT prone to Cardiovascular Disease")
            
        else:
            st.write(name,"-- Based on your inputs, you are prone to Cardiovascular Disease.")
            st.write("For More Information on Cardiovascular Disease, please visit " 
            "[CDC: Heart Disease Facts] (https://www.cdc.gov/heartdisease/facts.htm) and [Cardiovascular Disease Prevention] (https://www.cdc.gov/heartdisease/prevention.htm)")


st.sidebar.title("About")
st.sidebar.info('This app was created by: '

                    'Brittney Portes, Diana Estrada and Jenny Fernandez '
                    'using Streamlit to display various Models '
                    'for Cardiovascular Disease.')
