#pip install streamlit
#pip install pandas
#pip install sklearn

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json


# Load the dataset
df = pd.read_csv('c:\\Users\\Chaymae\\Desktop\\diabetes_cleaned_v2.csv')



st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# PATIENT DATA
def user_report():
    Pregnancies = st.slider("Your Number of Pregnancies", 0, 17, 3)
    Glucose = st.slider("Your Glucose", 0, 200, 120)
    BloodPressure = st.slider("Your Blood Pressure", 0, 122, 70)
    SkinThickness = st.slider("Your Skin thickness", 0, 100, 20)
    Insulin = st.slider("Your Insulin", 0, 846, 79)
    BMI = st.slider("Your BMI", 0, 67, 20)
    DiabetesPedigreeFunction = st.slider("Your Diabetes Pedigree Function", 0.0, 2.4, 0.47)
    Age = st.slider("Your Age", 21, 88, 33)

    user_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    return user_data

user_data = user_report()


# MODEL
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sidebar
classifier_name = st.sidebar.selectbox("Select Classifier", ["Pregnancy", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "DPF"])

# Set color based on classifier_name (you can customize this part)
color_dict = {
    "Pregnancy": "green",
    "Glucose": "magenta",
    "Blood Pressure": "red",
    "Skin Thickness": "blue",
    "Insulin": "orange",
    "BMI": "purple",
    "DPF": "yellow"
}

color = color_dict.get(classifier_name, "green")  # Default to green if not found

# Plot based on the selected classifier_name
if classifier_name == "Pregnancy":
    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
    ax2 = sns.scatterplot(x='Age', y='Pregnancies', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    st.pyplot(fig_preg)

elif classifier_name == "Glucose":
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
    ax4 = sns.scatterplot(x='Age', y='Glucose', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    st.pyplot(fig_glucose)

elif classifier_name == "Blood Pressure":
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
    ax6 = sns.scatterplot(x='Age', y='BloodPressure', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    st.pyplot(fig_bp)

elif classifier_name == "Skin Thickness":
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
    ax8 = sns.scatterplot(x='Age', y='SkinThickness', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 110, 10))
    st.pyplot(fig_st)

elif classifier_name == "Insulin":
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
    ax10 = sns.scatterplot(x='Age', y='Insulin', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 900, 50))
    st.pyplot(fig_i)

elif classifier_name == "BMI":
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax11 = sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
    ax12 = sns.scatterplot(x='Age', y='BMI', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 70, 5))
    st.pyplot(fig_bmi)

elif classifier_name == "DPF":
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax13 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
    ax14 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=user_data, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 3, 0.2))
    st.pyplot(fig_dpf)


# Close the matplotlib figures to avoid displaying extra plots
plt.close('all')


st.title('Diabetes Prediction')
st.sidebar.header('Patient Data')
st.sidebar.write(user_data)


if st.button('Predict'):
    result = rf.predict(user_data)
    updated_res = result.flatten().astype(int)
    if updated_res == 0:
       st.write("You not are diabetic")
    else:
       st.write("You are diabetic")
   

# Lottie Animation
from streamlit_lottie import st_lottie
import json

# Example Lottie animation from a local file
animation_file_path = r"C:\Users\Chaymae\Downloads\opener-loading.json"

# Function to load Lottie animation from file
def load_lottie_file(file_path):
    with open(file_path, "r") as file:
        animation_json = json.load(file)
    return st_lottie(animation_json, height=300, key="user")

# Display the Lottie animation
load_lottie_file(animation_file_path)
