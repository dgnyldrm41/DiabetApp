# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:17:59 2024

@author: dogan.yildirim
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
import streamlit as st
import streamlit.components.v1 as components

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE

# Veri setini yükleme
df = pd.read_csv("diabetes.csv")

# Eksik değerleri NaN ile değiştirme
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Eksik değerleri KNNImputer ile doldurma
imputer = KNNImputer(n_neighbors=5)
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Önemli özellikleri seçme
important_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df_selected = df_filled[important_features]

# Sınıf dengesizliğini yönetmek için SMOTE kullanma
X = df_selected.drop("Outcome", axis=1)
y = df_selected['Outcome']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test bölümlemesi
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Modellerin tanımlanması
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Naive Bayes": GaussianNB()
}

# En iyi modeli bulmak için model performansını değerlendirme
best_model = None
best_recall = 0
best_accuracy = 0

for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    if recall > best_recall:
        best_model = model
        best_recall = recall
        best_accuracy = accuracy



# Web uygulamasını oluşturma

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<h1 style='text-align: Left; color: green;'>Diabetes Dataset Prediction App</h1>", unsafe_allow_html=True)
    st.image('data.jpg', caption='Miuul Data Analyst Bootcamp', width=350)

with col2:
    name = st.text_input('What is your name?').capitalize()

    def get_user_input():
        pregnancies = st.number_input("Enter Pregnancies")
        glucose = st.number_input('Enter Glucose')
        bldp = st.number_input("Enter BloodPresssure")
        skin_thickness = st.number_input('Enter Skin Thickness')
        insulin = st.number_input('Enter Insulin')
        BMI = st.number_input('Enter BMI')
        DPF = st.number_input('Enter DPF')
        age = st.number_input('Enter Age')

        user_data = {'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': bldp,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': BMI,
                    'DiabetesPedigreeFunction': DPF,
                    'Age': age
                    }
        features = pd.DataFrame(user_data, index=[0])
        return features

    user_input = get_user_input()

    st.header("Correlation Heatmap")
    fig_heatmap = plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig_heatmap)

    # st.header("Pairplot of Features")
    # fig_pairplot = sns.pairplot(df, hue='Outcome', palette='coolwarm')
    # st.pyplot(fig_pairplot)

    # Buton ile tahmin sonucunu almak

with col3:
    bt = st.button('Get Result')

    if bt:
        try:
            # Kullanıcının girdiği verilere göre tahmin yapma
            test_result = best_model.predict(user_input)
            st.session_state['test_result'] = test_result
            st.session_state['name'] = name

            if test_result == 1:
                st.session_state['diabetes'] = True

            else:
                st.session_state['diabetes'] = False

        except:
            st.warning("Please fill all the required information.")

    # Tahmin sonucunu gösterme
    if 'diabetes' in st.session_state:
        if st.session_state['diabetes']:
            st.write(st.session_state['name'], ":red[You have diabetes. You must eat less :)]")

        else:
            st.write(st.session_state['name'], ":green[You don't have Diabetes. You can eat more :)]")

    # Görsel analizler
    st.header('Visual Analysis')

    # Görsel analizlerin devamı
    if 'diabetes' in st.session_state:
        test_result = best_model.predict(user_input)

        if test_result == 1:
            color = 'red'
        else:
            color = 'green'

        # Scatter plotlar ile görsel analiz
        for feature in X.columns:
            if feature != 'Outcome':
                fig = plt.figure()
                sns.scatterplot(x='Age', y=feature, hue='Outcome', data=df_selected, palette='coolwarm')
                plt.scatter(user_input['Age'], user_input[feature], color=color, s=100, marker='o', label='Your Data')
                plt.title(f'Age vs {feature}')
                plt.legend()
                st.pyplot(fig)

# Ölüm oranları verilerini yükleme
df2 = pd.read_csv("death.csv")

# Sonuçları gösterme
with col4:
    if 'diabetes' in st.session_state:
        if st.session_state['diabetes']:
            st.write(st.session_state['name'], ":red[You have diabetes. You must eat less :)]")

            # Ölüm oranları için seçim yapma
            country = st.selectbox('Select your country:', df2['Entity'].unique(), key='country')
            year = st.selectbox('Select year:', df2['Year'].unique(), key='year')

            if country and year:
                death_rate = df2[(df2['Entity'] == country) & (df2['Year'] == year)]['Deaths'].iloc[0]
                st.write(f"In {country}, in {year} the average death rate due to diabetes is {death_rate:.2f}%.")

        else:
            st.write(st.session_state['name'], ":green[You don't have Diabetes. You can eat more :)]")

    # Model performans metrikleri
    if 'diabetes' in st.session_state:
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.subheader('Model Performance Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')
        st.write(f'Best Model: {best_model}')
        st.write(f'Best Recall: {best_recall:.2f}')
        st.write(f'Best Accuracy: {best_accuracy:.2f}')