import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Title
st.title('PandasAI Data Explorer')

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Instantiate a LLM
    llm = OpenAI(api_token="OPENAI_API_KEY")
    df = SmartDataframe(df, config={"llm": llm})
    
    # Display the DataFrame
    st.write(df.head())
    
    # Chat with the DataFrame
    user_input = st.text_input("Ask a question to your data:")
    if user_input:
        response = df.chat(user_input)
        st.write(response)
