import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

def load_data():
    df = pd.read_csv('Data Greenie All - Sheet 1.csv')
    return df

# Load the data
data = load_data()

st.set_page_config(page_title="ESG Interactive AI Dashboard", page_icon=":earth_asia:",layout="wide")
st.title('🌏 ESG Interactive AI Dashboard')
st.dataframe(data)

#data.drop('index',axis=1)
#options = ['Environment', 'Social', 'Governance']

#selected_option = st.selectbox('Select an ESG Pillar Score:', options)

#st.write('You selected:', selected_option)

import matplotlib.pyplot as plt

# demo1
st.title('ESG Pillar Score demo1')
selected_column = st.selectbox('Select an ESG Pillar Score', data.columns)
selected_data = data[selected_column]
st.bar_chart(selected_data)

#demo2
st.title('ESG Pillar Score demo2')
company = st.selectbox('Select a company', data['Environmental Pillar Score_2018'])
df_filtered = data[data['Environmental Pillar Score_2018'] == company]
st.line_chart(df_filtered['Stock Name'], use_container_width=True)

# demo3
#st.title('ESG Pillar Score demo3')
#selected_column = st.selectbox('Select an ESG Pillar Score', data['Environmental Pillar Score_2018'])
#selected_data = data[selected_column]
#st.bar_chart(selected_data)

# demo4
st.title('ESG Pillar Score demo4')
options = ['Environmental Pillar Score_2018', 'Environmental Pillar Score_2019', 'Environmental Pillar Score_2020']
selected_columns = st.multiselect('Select columns:', options)
selected_data = data[selected_columns]
st.bar_chart(selected_data)

#demo 5
st.title('ESG Pillar Score demo5')
options = ['Environmental Pillar Score_2018', 'Environmental Pillar Score_2019', 'Environmental Pillar Score_2020']
selected_columns = st.selectbox('Select columns:', options)
selected_data = data[selected_columns].head(10)
st.bar_chart(selected_data)

#demo6
st.title('Top 10 Largest Values')
options = ['Environmental Pillar Score_2018', 'Environmental Pillar Score_2019', 'Environmental Pillar Score_2020']
selected_columns = st.selectbox('Select columns:', options ,key='select_column')
top_10_largest = data.nlargest(10, selected_column)
st.write(top_10_largest)
st.bar_chart(top_10_largest[[selected_column]])

#demo7
st.title('ESG Pillar Score demo7')
options = ['Environmental Pillar Score_2018', 'Environmental Pillar Score_2019', 'Environmental Pillar Score_2020']
selectbox_key = 'select_column_' + str(hash(options))
selected_columns = st.selectbox('Select columns:', options, key='select_column')
selected_data = data.nlargest(10, selected_columns)
st.bar_chart(selected_data[selected_columns])
