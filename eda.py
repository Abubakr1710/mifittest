import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image

image = Image.open('mi.png')
st.image(image)

st.title('Hello and Welcome to our Project')
st.write('This project will try to find best model to classify the target and calculate calories')


df_sec = pd.read_csv('dataset_5secondWindow.csv')
st.write('Users by target')
st.write(plt.figure(figsize=(25, 12)),sns.countplot(x='user', hue='target',data=df_sec.sort_values(by=['user'])),plt.legend(loc='upper right'))
#st.write(plt.figure(figsize=(25,12)), sns.histplot(x=df_sec["user"], hue=df_sec["target"], palette="pastel", color='b'))
image = Image.open('sound vs targets.PNG')
st.write('Sound vs target')
st.image(image)
image = Image.open('speed vs target.PNG')
st.write('Speed vs target')
st.image(image)

st.write('Target by time')
st.write(plt.figure(figsize=(25,12)),sns.histplot(x=df_sec['time'],color = df_sec['target'],hue=df_sec["target"]))
