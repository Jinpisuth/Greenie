import streamlit as st
import pandas as pd

def load_data():
    df = pd.read_csv('Data Greenie All - Sheet 1.csv')
    return df

# Load the data
data = load_data()

Data Greenie All - Sheet 1.csv
st.set_page_config(page_title="ESG Analysis Dashboard", page_icon=":earth_asia:",layout="wide")
st.title('🌏 ESG Interactive AI Dashboard')
