from email import header
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('Hello and Welcome to our Project')

dataset_name = st.sidebar.selectbox('Select Dataset:',('5 Second', 'Half Second'))
st.write('Dataset Name:', dataset_name)

classifier_name = st.sidebar.selectbox('Select Classifier:',{'KNN','SVM','Extra Trees', 'Decision Tree', 'Random Forest'})
df1 = pd.read_csv('dataset_5secondWindow.csv')
df2 = pd.read_csv('dataset_halfSecondWindow.csv')

df1_filled = df1.fillna(-999)
df2_filled = df2.fillna(-999)

df1_filled = pd.get_dummies(df1_filled)
df2_filled = pd.get_dummies(df2_filled)


def get_data(dataset_name):
    data = None
    if dataset_name == "5 Second":
        data = df1_filled
        st.write(df1_filled.iloc[:,-17:-13])
    else:
        data = df2_filled
        st.write(df2_filled.iloc[:,-17:-13])

    X = data.drop(['target_Car','target_Still','target_Train','target_Walking'], axis = 1)
    y = data.iloc[:,-17:-13]
    return X, y  
X, y = get_data(dataset_name)
st.write('Shape of Dataset', X.shape)

def add_param_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 40)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_debth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1,100)
        params['max_debth'] = max_depth
        params['n_estimators'] = n_estimators
    elif clf_name =='Decision Tree':
        max_depth = st.sidebar.slider('max_debth', 2, 15)
        params['max_debth'] = max_depth
    elif clf_name == 'Extra Trees':
        max_depth = st.sidebar.slider('max_debth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1,100)
        params['max_debth'] = max_depth
        params['n_estimators'] = n_estimators


    return params
params = add_param_ui(classifier_name)


def get_classifier(clf_name,params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C= params['C'])
    elif clf_name == 'Random Forest':
        clf  = RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_debth'],random_state=101) 
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(max_depth=params['max_debth'],random_state=101)
    elif clf_name == 'Extra Trees':
        clf = ExtraTreesClassifier(n_estimators=params['n_estimators'],max_depth=params['max_debth'],random_state=101)

    return clf  

clf = get_classifier(classifier_name, params)


#----------------------------------------------------------------------#
#Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
st.write(f"Classifier  : {classifier_name}")
st.write(f'Accuracy : {acc}')





