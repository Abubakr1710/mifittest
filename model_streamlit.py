from email import header
from time import time
import streamlit as st
import pandas as pd
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OrdinalEncoder
from numpy import asarray
all  = st.container()
with all:
   
 dataset_name = st.sidebar.selectbox('Select Dataset:',('5 Second', 'Half Second'))
 st.write('Dataset Name:', dataset_name)

# classifier_name = st.sidebar.selectbox('Select Classifier:',{'KNN','Extra Trees', 'Decision Tree', 'Random Forest'})
# df1 = pd.read_csv('dataset_5secondWindow.csv')
# df2 = pd.read_csv('dataset_halfSecondWindow.csv')

# df1_filled = df1.fillna(0)
# df2_filled = df2.fillna(0)

# oi = OrdinalEncoder()
# cd = asarray(df1_filled['user'])
# df1_filled['user'] = oi.fit_transform(cd.reshape(-1,1))
# df1_filled = df1_filled.sort_values(by='user')
# df1_filled = df1_filled.drop(['Unnamed: 0','id','activityrecognition#0','user'], axis=1)
# cd = asarray(df1_filled['target'])
# df1_filled['target'] = oi.fit_transform(cd.reshape(-1,1))



# oi = OrdinalEncoder()
# cd = asarray(df2_filled['user'])
# df2_filled['user'] = oi.fit_transform(cd.reshape(-1,1))
# df2_filled = df2_filled.sort_values(by='user')
# df2_filled = df2_filled.drop(['Unnamed: 0','id','activityrecognition#0','user'], axis=1)
# cd = asarray(df2_filled['target'])
# df2_filled['target'] = oi.fit_transform(cd.reshape(-1,1))



# def get_data(dataset_name):
#     data = None
#     if dataset_name == "5 Second":
#         data = df1_filled
#         data_test = data.iloc[:2500, :]
#         data_train = data.iloc[2500: , :]
#         st.write(df1_filled)
#     else:
#         data = df2_filled
#         st.write(df2_filled)
#         data_test = data.iloc[:25000, :]
#         data_train = data.iloc[25000: , :]

#     X = data.drop(['target'], axis=1)
#     y = data['target']

#     X_train = data_train.iloc[:,:66]
#     y_train = data_train.iloc[:, 66:]

#     ### Testing sets
#     X_test = data_test.iloc[:, :66]
#     y_test = data_test.iloc[:, 66:]

#     return X, y, data_test, data_train, X_train, y_train, X_test, y_test  
# X, y, data_test, data_train, X_train, y_train, X_test, y_test = get_data(dataset_name)
# st.write('Shape of Dataset', X.shape)

# def add_param_ui(clf_name):
#     params = dict()
#     if clf_name == 'KNN':
#         K = st.sidebar.slider('K', 1, 40)
#         params['K'] = K
#     elif clf_name == 'Random Forest':
#         max_depth = st.sidebar.slider('max_debth', 2, 15)
#         n_estimators = st.sidebar.slider('n_estimators', 1,100)
#         params['max_debth'] = max_depth
#         params['n_estimators'] = n_estimators
#     elif clf_name =='Decision Tree':
#         max_depth = st.sidebar.slider('max_debth', 2, 15)
#         params['max_debth'] = max_depth
#     elif clf_name == 'Extra Trees':
#         max_depth = st.sidebar.slider('max_debth', 2, 15)
#         n_estimators = st.sidebar.slider('n_estimators', 1,100)
#         params['max_debth'] = max_depth
#         params['n_estimators'] = n_estimators


#     return params
# params = add_param_ui(classifier_name)


# def get_classifier(clf_name,params):
#     if clf_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     elif clf_name == 'Random Forest':
#         clf  = RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_debth'],random_state=101) 
#     elif clf_name == 'Decision Tree':
#         clf = DecisionTreeClassifier(max_depth=params['max_debth'],random_state=101)
#     elif clf_name == 'Extra Trees':
#         clf = ExtraTreesClassifier(n_estimators=params['n_estimators'],max_depth=params['max_debth'],random_state=101)

#     return clf  

# clf = get_classifier(classifier_name, params)


# #----------------------------------------------------------------------#
# #Classification
# start_time = time.time()

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# total_time = (time.time() - start_time).__round__(3)

# acc  = accuracy_score(y_test, y_pred).round(2)*100
# bal_acc = balanced_accuracy_score(y_test, y_pred).round(2)*100


# # st.write(f"Model Name  : {classifier_name}")
# # st.write(f'Accuracy : {acc}')
# # st.write(f'Balanced Accuracy: {bal_acc}')
# # st.write(f'Time: {total_time}')


# #----------------------------------------------------------------------#
# tree_classifiers= {
#   "Decision Tree": DecisionTreeClassifier(),
#   "Extra Trees":ExtraTreesClassifier(),
#   "Random Forest": RandomForestClassifier(),
#   #"KNN": KNeighborsClassifier()
#   #"AdaBoost":AdaBoostClassifier(),
#   #"Skl GBM":GradientBoostingClassifier(),
#   #"XGBoost":XGBClassifier(),
#   #"LightGBM":LGBMClassifier(),
#   #"CatBoost":CatBoostClassifier()
#    }

# tree_classifiers = {name: make_pipeline(model) for name, model in tree_classifiers.items()}

# results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})


# rang = abs(y_train.max()) - abs(y_train.min())
# for model_name, model in tree_classifiers.items():
    
#     start_time = time.time()
#     model.fit(X_train, y_train)
#     total_time = time.time() - start_time
        
#     pred = model.predict(X_test)
    
#     results = results.append({"Model":    model_name,
#                               "Accuracy": accuracy_score(y_test, pred)*100,
#                               "Bal Acc.": balanced_accuracy_score(y_test, pred)*100,
#                               "Time":     total_time},
#                               ignore_index=True)
                              
                              
# results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
# results_ord.index += 1 
# results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='Blues')

# st.write(results)
