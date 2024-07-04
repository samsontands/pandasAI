import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os

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
        
        llm_type = st.selectbox("Please select LLM", ('BambooLLM', 'gemini-pro'), index=0)

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df])

        llm = get_LLM(llm_type)

        if llm:
            # Instantiating PandasAI agent
            analyst = get_agent(data, llm)

            # Starting the chat with the PandasAI agent
            chat_window(analyst)
    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

def get_LLM(llm_type):
    try:
        if llm_type == 'BambooLLM':
            api_key = st.secrets["PANDASAI_API_KEY"]
            llm = BambooLLM(api_key=api_key)
        elif llm_type == 'gemini-pro':
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
        return llm
    except Exception as e:
        st.error("Error accessing API key from Streamlit secrets. Please ensure the key is properly set.")
        return None

def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explore your data with PandasAI?🧐")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message['response'])
            elif 'error' in message:
                st.text(message['error'])

    user_question = st.chat_input("What are you curious about? ")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "question": user_question})
       
        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "response": response})
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

    def clear_chat_history():
        st.session_state.messages = []

    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def get_agent(data, llm):
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return agent

def extract_dataframes(raw_file):
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df
    elif raw_file.name.split('.')[1] in ['xlsx', 'xls']:
        xls = pd.ExcelFile(raw_file)
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)
    return dfs

if __name__ == "__main__":
    main()
