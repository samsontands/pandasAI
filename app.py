import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import base64
from io import BytesIO

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
        # Activating Demo Data
        st.text("Data Setup: 📝$2a$10$HfRGtOlM.Mp11kh76ipw9.qdScO0nwyVSQeozauNAa.SIfZtTQCmm")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])

        st.markdown(":green[*Please ensure the first row has the column names.*]")

        # Selecting LLM to use
        llm_type = st.selectbox(
            "Please select LLM",
            ('BambooLLM', 'gemini-pro'), index=0)
        
        # Adding users API Key
        user_api_key = st.text_input('Please add your API key', placeholder='Paste your API key here', type='password')
        
        # Get Pandas API key here
        st.markdown("[Get Your PandasAI API key here](https://www.pandabi.ai/auth/sign-up)")

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!",
                          tuple(data.keys()), index=0
                          )
        st.dataframe(data[df])

        llm = get_LLM(llm_type, user_api_key)

        if llm:
            # Instantiating PandasAI agent
            analyst = get_agent(data, llm)

            # Starting the chat with the PandasAI agent
            chat_window(analyst)
    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

def get_LLM(llm_type, user_api_key):
    try:
        if llm_type == 'BambooLLM':
            if user_api_key:
                os.environ["PANDASAI_API_KEY"] = user_api_key
            else:
                # If no API key provided, try to get it from environment variables
                os.environ["PANDASAI_API_KEY"] = os.getenv('PANDASAI_API_KEY')
            llm = BambooLLM()
        elif llm_type == 'gemini-pro':
            if user_api_key:
                genai.configure(api_key=user_api_key)
            else:
                # Configure the API key
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=user_api_key)
        return llm
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explore your data with PandasAI?🧐")

    # Initializing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Displaying the message history on re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message['response'])
            elif 'error' in message:
                st.text(message['error'])

    # Getting the questions from the users
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
                
                # Display and offer downloads for any generated charts
                display_and_download_charts()
        
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"
            st.error(error_message)

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []
    # Button for clearing history
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def get_agent(data, llm):
    agent = Agent(
        list(data.values()),
        config={
            "llm": llm,
            "verbose": True,
            "response_parser": StreamlitResponse,
            "save_charts": True,  # Enable saving of generated charts
            "save_charts_path": "generated_charts"  # Specify the directory to save charts
        }
    )
    return agent

def display_and_download_charts():
    chart_dir = "generated_charts"
    if os.path.exists(chart_dir) and os.listdir(chart_dir):
        st.subheader("Generated Charts")
        for chart_file in os.listdir(chart_dir):
            with open(os.path.join(chart_dir, chart_file), "rb") as file:
                btn = st.download_button(
                    label=f"Download {chart_file}",
                    data=file,
                    file_name=chart_file,
                    mime="image/png"
                )
            st.image(os.path.join(chart_dir, chart_file))
    else:
        st.info("No charts have been generated yet.")

def extract_dataframes(raw_file):
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df
    elif (raw_file.name.split('.')[1] == 'xlsx') or (raw_file.name.split('.')[1] == 'xls'):
        xls = pd.ExcelFile(raw_file)
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)
    return dfs

if __name__ == "__main__":
    main()
