import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def load_data():
    df = pd.read_csv('Data Greenie All - Sheet 1.csv')
    return df

# Load the data
data = load_data()

st.set_page_config(page_title="ESG Interactive AI Dashboard", page_icon=":earth_asia:",layout="wide")
st.title('🌏 ESG Interactive AI Dashboard')
st.dataframe(data)

