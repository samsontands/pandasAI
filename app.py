import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Initialize the OpenAI LLM
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(api_key=openai_api_key)

# Initialize PandasAI
pandas_ai = PandasAI(llm)

# Streamlit app layout
st.title("PandasAI with Streamlit")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame Preview:", df.head())
    
    user_query = st.text_input("Ask a question about your data:")
    
    if st.button("Submit"):
        response = pandas_ai(df, user_query)
        st.write("Response:", response)
