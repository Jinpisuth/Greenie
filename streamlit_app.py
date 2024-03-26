name: Python package

on: [push]

jobs:
  build:

    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
import streamlit as st
import pandas as pd
#import plotly.express as px
import numpy as np

st.set_page_config(page_title="ESG Interactive AI Dashboard", page_icon=":earth_asia:",layout="wide")
st.title('🌏 ESG Interactive AI Dashboard')

def load_data():
    df = pd.read_csv('Data Greenie All - Sheet1.csv')
    return df

data = load_data()
st.title('🚀 Welcome to ESG Investments Dashboard with Gemini AI-Chatbot Analysis')

welcome_text = (
    "In today's financial world, it's becoming more important to consider "
    "Environmental, Social, and Governance (ESG) factors when investing. "
    "Our team 'Greenie' created a special tool called an ESG Investments Dashboard "
    "with Gemini AI-Chatbot Analysis. This dashboard brings together important "
    "information about how companies are doing in terms of ESG, leveraging our "
    "Gemini AI Chatbot for analysis. Utilizing the technology such as Natural "
    "Language Processing (NLP) and Data Visualization, we aim for user-friendly "
    "comprehension. With this dashboard, investors can make better choices that "
    "match their goals for a better world."
    "This project is developed by Greenie Team for QuantCorner-LSEG Hackathon: the Quest for Sustainable Alpha"
)

st.write(welcome_text)
st.write('—' * 80)
# Display the data using Streamlit
#st.title('ESG Data')
#st.write('Here is the imported data:')
#st.dataframe(data)

def load_data():
    df = pd.read_csv('Data Greenie All - Sheet1.csv') 
    return df 
df = load_data()
#sidebar
st.sidebar.title("Parameters")

S = st.sidebar.selectbox('Select a Stock name', df['Stock Name']), 
                     

X = st.sidebar.selectbox('Sector', df['Sector'])

# Filter data
filtered_df1 = df[df['Stock Name'] == S]
filtered_df2 = df[df['Sector'] == X]
# Display the filtered data
#st.write(filtered_df1,filtered_df2)
#data.drop('index',axis=1)
#options = ['Environment', 'Social', 'Governance']

#selected_option = st.selectbox('Select an ESG Pillar Score:', options)

#st.write('You selected:', selected_option)

import matplotlib.pyplot as plt

# demo1
st.title('ESG Pillar Score')
selected_column = st.selectbox('Select an ESG Pillar Score', data.columns)
selected_data = data[selected_column]
st.bar_chart(selected_data)

#demo2
#st.title('ESG Pillar Score demo2')
#company = st.selectbox('Select a company', data['Environmental Pillar Score_2019'])
#df_filtered = data[data['Environmental Pillar Score_2019'] == company]
#st.line_chart(df_filtered['Stock Name'], use_container_width=True)

# demo3
#st.title('ESG Pillar Score demo3')
#selected_column = st.selectbox('Select an ESG Pillar Score', data['Environmental Pillar Score_2018'])
#selected_data = data[selected_column]
#st.bar_chart(selected_data)

# demo4
#st.title('ESG Pillar Score demo4')
#options = ['Environmental Pillar Score_2019', 'Environmental Pillar Score_2020', 'Environmental Pillar Score_2021']
#selected_columns = st.multiselect('Select columns:', options)
#selected_data = data[selected_columns]
#st.bar_chart(selected_data)

#demo 5
#st.title('ESG Pillar Score demo5')
#options = ['Environmental Pillar Score_2019', 'Environmental Pillar Score_2020', 'Environmental Pillar Score_2021']
#selected_columns = st.selectbox('Select columns:', options)
#selected_data = data[selected_columns].head(10)
#st.bar_chart(selected_data)

#demo6
st.title('Top 10 Largest Values')
options = ['Environmental Pillar Score_2019', 'Environmental Pillar Score_2020', 'Environmental Pillar Score_2021']
selected_columns = st.selectbox('Select columns:', options ,key='select_column')
top_10_largest = data.nlargest(10, selected_column)
#st.write(top_10_largest)
st.bar_chart(top_10_largest[[selected_column]])


#env scatter 2
st.title("envirionment Scatter")
options = ['Governance Pillar Score_2022', 'Environmental Pillar Score_2022', 'Social Pillar Score_2022']
column_x = st.selectbox('Select column for y-axis', options)

b1,b2=st.columns(2)
#dt = pd.DataFrame(data) 
#b1.dataframe(dt,use_container_width=True)


with b2:
  #plt.figure(figsize=(8, 6))
 plt.scatter(df[column_x], df['Total Return_2024'], label='Total Return')
 #plt.plot(df[selected_column], model.predict(X), color='red', label='Predicted')
 plt.ylabel(column_x)
 plt.xlabel('Total Return')
 plt.title(f'Line of Best Fit ({selected_column} vs Projects)')
 plt.title(f'Line of Best Fit ({selected_column} vs Projects)')
 plt.grid(color='grey', linestyle='--') 
 #plt.legend()


 plt.gca().spines['top'].set_color('gray')
 plt.gca().spines['bottom'].set_color('gray')
 plt.gca().spines['left'].set_color('gray')
 plt.gca().spines['right'].set_color('gray')
 st.pyplot(plt)

#table
st.title('ESG Data')
st.write('Here is the imported data:')
st.dataframe(data)

#Gemini
#API_Key="AIzaSyDWv8ac5_OIxx5dOZA8HkKH2nyVMN-bJO8"

from PIL import Image
import io
import os
import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm

with st.sidebar:
    st.title("Gemini API")
    api_key = os.getenv("GOOGLE_API_KEY")
    api_key = st.text_input("API key")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        if "api_key" in st.secrets:
            genai.configure(api_key=st.secrets["api_key"])
            st.success('API key already provided!', icon='✅')
            api_key = st.secrets['GOOGLE_API_KEY']
        else:
            api_key = st.text_input('Enter Google API Key:', type='password')
            if not (api_key.startswith('AI')):
              st.warning('Please enter your API Key!', icon='⚠️')
            else:
              st.success('Success!', icon='✅')
    os.environ['GOOGLE_API_KEY'] = api_key
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/wms31/streamlit-gemini/blob/main/app.py)"
    "[Check out the blog post!](https://letsaiml.com/creating-google-gemini-app-with-streamlit/)"

    select_model =["gemini-pro-vision"]
    if select_model == "gemini-pro-vision":
        uploaded_image = st.file_uploader(
            "upload image",
            label_visibility="collapsed",
            accept_multiple_files=False,
            type=["png", "jpg"],
        )
        st.caption(
            "Note: The vision model gemini-pro-vision is not optimized for multi-turn chat."
        )
        if uploaded_image:
            image_bytes = uploaded_image.read()

def get_response(messages, model="gemini-pro"):
    model = genai.GenerativeModel(model)
    res = model.generate_content(messages, stream=True,
                                safety_settings={'HARASSMENT':'block_none'})
    return res

if "messages" not in st.session_state:
    st.session_state["messages"] = []
messages = st.session_state["messages"]

# The vision model gemini-pro-vision is not optimized for multi-turn chat.
#if messages and select_model != "gemini-pro-vision":
 #   for item in messages:
  #      role, parts = item.values()
   #     if role == "user":
    #        st.chat_message("user").markdown(parts[0])
     #   elif role == "model":
      #      st.chat_message("assistant").markdown(parts[0])

#chat_message = st.chat_input("Say something")
#generate_t2t = st.button("Generate my travel itinerary", key="generate_t2t")

with st.container():
  destination_name = st.text_input("Stock Name: \n\n",value="MINT.BK")
  days = st.text_input("Average of ESG Score",value="89.815")
  suggested_attraction = st.text_input("Total Return",value="0.8547")
  config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
    
  generate_t2t = st.button("Generate", key="generate_t2t")
  model = genai.GenerativeModel("gemini-pro", generation_config=config)
  if generate_t2t:
      with st.spinner("Generating your travel itinerary using Gemini..."):
        if response:
          st.write("Your plan:")
          st.write(response.text)
