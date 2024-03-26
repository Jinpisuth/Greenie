import streamlit as st
import pandas as pd
#import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ESG Interactive AI Dashboard", page_icon=":earth_asia:",layout="wide")
st.title('🌏 ESG Interactive AI Dashboard')

def load_data():
    df = pd.read_csv('Data Greenie All - Sheet 1.csv')
    return df

# Load the data
data = load_data()

# Display the data using Streamlit
st.title('My Streamlit App')
st.write('Here is the imported data:')
st.dataframe(data)

#data.drop('index',axis=1)
#options = ['Environment', 'Social', 'Governance']

#selected_option = st.selectbox('Select an ESG Pillar Score:', options)

#st.write('You selected:', selected_option)

# demo1
st.title('ESG Pillar Score demo1')
selected_column = st.selectbox('Select an ESG Pillar Score', data.columns)
selected_data = data[selected_column]
st.bar_chart(selected_data)

#demo2
st.title('ESG Pillar Score demo2')
company = st.selectbox('Select a company', data['Environmental Pillar Score_2019'])
df_filtered = data[data['Environmental Pillar Score_2019'] == company]
st.line_chart(df_filtered['Stock Name'], use_container_width=True)

# demo3
#st.title('ESG Pillar Score demo3')
#selected_column = st.selectbox('Select an ESG Pillar Score', data['Environmental Pillar Score_2018'])
#selected_data = data[selected_column]
#st.bar_chart(selected_data)

# demo4
st.title('ESG Pillar Score demo4')
options = ['Environmental Pillar Score_2019', 'Environmental Pillar Score_2020', 'Environmental Pillar Score_2021']
selected_columns = st.multiselect('Select columns:', options)
selected_data = data[selected_columns]
st.bar_chart(selected_data)

#demo 5
st.title('ESG Pillar Score demo5')
options = ['Environmental Pillar Score_2019', 'Environmental Pillar Score_2020', 'Environmental Pillar Score_2021']
selected_columns = st.selectbox('Select columns:', options)
selected_data = data[selected_columns].head(10)
st.bar_chart(selected_data)

#demo6
st.title('Top 10 Largest Values')
options = ['Environmental Pillar Score_2019', 'Environmental Pillar Score_2020', 'Environmental Pillar Score_2021']
selected_columns = st.selectbox('Select columns:', options ,key='select_column')
top_10_largest = data.nlargest(10, selected_column)
st.write(top_10_largest)
st.bar_chart(top_10_largest[[selected_column]])




#demo8
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.metric_cards import style_metric_cards 
 
st.title('ESG Pillar Score demo8')
style_metric_cards(background_color="#FFFFFF",border_left_color="#686664")

#navicon and header
#st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")  

st.header("COVARIANCE FOR TWO RONDOM VARIABLES")
st.success("The main objective is to measure if Number of family Environment or Social may influence a person to supervise many Governance")
 
# load CSS Style
#with open('style.css')as f:
    #st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

def load_data():
    return pd.read_csv('/content/Data Greenie All - Sheet1.csv')
df = load_data()
selected_column = st.selectbox('SELECT INPUT X FEATURE', df.select_dtypes("number").columns)
X = sm.add_constant(df[selected_column])  # Adding a constant for intercept

# Fitting the model
model = sm.OLS(df['Book Value Percentage of Market Capitalization_2022'],X).fit()


c1,c2,c3,c4=st.columns(4)
# Printing general intercept
c1.metric("INTERCEPT:",f"{model.params[0]:,.4f}")

# Printing R-squared
c2.metric("R SQUARED",f"{model.rsquared:,.2f}",delta="is it strong relationship ?")

# Printing adjusted R-squared
c3.metric("ADJUSTED R",f"{model.rsquared_adj:,.3f}",)

# Printing standard error
c4.metric("STANDARD ERROR",f"{model.bse[0]:,.4f}")

# Printing correlation coefficient
 

style_metric_cards(background_color="#FFFFFF",border_left_color="#686664")

b1,b2=st.columns(2)
# Printing predicted values
data = {
    'X feature':selected_column,
    'Prediction': model.predict(X),
    'Residuals':model.resid
}

dt = pd.DataFrame(data) 
b1.dataframe(dt,use_container_width=True)

with b2:
 plt.figure(figsize=(8, 6))
 plt.scatter(df[selected_column], df['Book Value Percentage of Market Capitalization_2022'], label='Actual')
 plt.plot(df[selected_column], model.predict(X), color='red', label='Predicted')
 plt.xlabel(selected_column)
 plt.ylabel('Projects')
 plt.title(f'Line of Best Fit ({selected_column} vs Projects)')
 plt.title(f'Line of Best Fit ({selected_column} vs Projects)')
 plt.grid(color='grey', linestyle='--') 
 plt.legend()

 # Setting outer border color
 plt.gca().spines['top'].set_color('gray')
 plt.gca().spines['bottom'].set_color('gray')
 plt.gca().spines['left'].set_color('gray')
 plt.gca().spines['right'].set_color('gray')
 st.pyplot(plt)

#demo 8
st.sidebar.title("Parameters")

S = st.sidebar.selectbox('Select a Stock name', df['Stock Name']), 
                     

X = st.sidebar.selectbox('Sector', df['Sector'])

# Filter data
filtered_df1 = df[df['Stock Name'] == S]
filtered_df2 = df[df['Sector'] == X]
# Display the filtered data
st.write(filtered_df1,filtered_df2)



#company = st.selectbox('Select a Stock name', data['Stock Name'])
#df_filtered = data[data['Stock Name'] == company]

#demo9
import os
import streamlit as st
import google.generativeai as genai 
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI with the API key
genai.configure(api_key=api_key)

# Everything is accessible via the st.secrets dict:

st.write("DB username:", st.secrets["myuser"])
st.write("DB password:", st.secrets["abcdef"])
st.write("My cool secrets:", st.secrets["my_cool_secrets"]["things_i_like"])

# And the root-level secrets are also accessible as environment variables:

import os
st.write(
	"Has environment variables been set:",
	os.environ["myuser"] == st.secrets["myuser"])
# Check if the Google API key is provided in the sidebar
with st.sidebar:
    if 'GOOGLE_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        api_key = st.secrets['GOOGLE_API_KEY']
    else:
        api_key = st.text_input('Enter Google API Key:', type='abcdef')
        if not (api_key.startswith('AI')):
            st.warning('Please enter your API Key!', icon='⚠️')
        else:
            st.success('Success!', icon='✅')
    os.environ['GOOGLE_API_KEY'] = api_key
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/wms31/streamlit-gemini/blob/main/app.py)"
    #"[Check out the blog post!](https://letsaiml.com/creating-google-gemini-app-with-streamlit/)"

# Set the title and caption for the Streamlit app
st.title("🤖 Google Gemini Models")
st.caption("🚀 A streamlit app powered by Google Gemini")

# Create tabs for the Streamlit app
tab1, tab2 = st.tabs(["🌏 Generate Travel Plans - Gemini Pro", "🖼️ Visual Venture - Gemini Pro Vision"])

# Code for Gemini Pro model
with tab1:
    st.write("💬 Using Gemini Pro - Text only model")
    st.subheader("🌏 Generate travel itineraries")
    
    destination_name = st.text_input("Enter destination name: \n\n",key="destination_name",value="United Arab Emirates")
    days = st.text_input("How many days would you like the itinerary to be? \n\n",key="days",value="5")
    suggested_attraction = st.text_input("What should the first suggested attraction be for the trip? \n\n",key="suggested_attraction",value="Visiting Burj Khalifa in Dubai.")
        
    prompt = f"""Come up with a {days}-day itinerary for a trip to {destination_name}. The first suggested attraction should be {suggested_attraction}
    """
    
    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
    
    generate_t2t = st.button("Generate my travel itinerary", key="generate_t2t")
    model = genai.GenerativeModel("gemini-pro", generation_config=config)
    if generate_t2t and prompt:
        with st.spinner("Generating your travel itinerary using Gemini..."):
            plan_tab, prompt_tab = st.tabs(["Travel Itinerary", "Prompt"])
            with plan_tab: 
                response = model.generate_content(prompt)
                if response:
                    st.write("Your plan:")
                    st.write(response.text)
            with prompt_tab: 
                st.text(prompt)

# Code for Gemini Pro Vision model
with tab2:
    st.write("🖼️ Using Gemini Pro Vision - Multimodal model")
    st.subheader("🔮 Generate image to text responses")
    
    image_prompt = st.text_input("Ask any question about the image", placeholder="Prompt", label_visibility="visible", key="image_prompt")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    image = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit=st.button("Generate Response")

    if submit:
        model = genai.GenerativeModel('gemini-pro-vision')
        with st.spinner("Generating your response using Gemini..."):
            if image_prompt!="":
                response = model.generate_content([image_prompt,image])
            else:
                response = model.generate_content(image)
        response = response.text
        st.subheader("Gemini's response")
        st.write(response)
