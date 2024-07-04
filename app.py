import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os

# Load environment variables
load_dotenv()

# Dictionary to store the extracted dataframes
data = {}

def main():
    st.set_page_config(page_title="PandasAI", page_icon="🐼")
    st.title("Chat with Your Data using PandasAI:🐼")
    
    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
        st.markdown(":green[*Please ensure the first row has the column names.*]")
        
        # Adding user's API Key
        user_api_key = st.text_input('Please add your PandasAI API key', placeholder='Paste your API key here', type='password')
        
        # Get Pandas API key here
        st.markdown("[Get Your PandasAI API key here](https://www.pandabi.ai/auth/sign-up)")

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df])

        llm = get_LLM(user_api_key)

        if llm:
            # Instantiating PandasAI agent
            analyst = get_agent(data, llm)

            # Starting the chat with the PandasAI agent
            chat_window(analyst)

    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

def get_LLM(user_api_key):
    try:
        if user_api_key:
            os.environ["PANDASAI_API_KEY"] = user_api_key
        else:
            # If no API key provided, try to get it from environment variables
            os.environ["PANDASAI_API_KEY"] = os.getenv('PANDASAI_API_KEY')

        llm = BambooLLM()
        return llm
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

# The rest of the functions (chat_window, get_agent, extract_dataframes) remain unchanged

if __name__ == "__main__":
    main()
