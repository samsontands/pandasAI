import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm import OpenAI

# Set up the page
st.set_page_config(page_title="PandasAI Chat App", page_icon="🐼", layout="wide")
st.title("PandasAI Chat App")

# Initialize session state for storing the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(st.session_state.df.head())

# Initialize PandasAI with OpenAI
llm = OpenAI(api_token=st.secrets["openai_api_key"])
pandas_ai = PandasAI(llm)

# Chat interface
if st.session_state.df is not None:
    user_question = st.text_input("Ask a question about your data:")
    if user_question:
        with st.spinner("Thinking..."):
            try:
                answer = pandas_ai.run(st.session_state.df, prompt=user_question)
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to start chatting with your data.")

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info(
    "This app allows you to upload a CSV file and chat with your data using PandasAI. "
    "Upload your data, then ask questions about it in natural language."
)

# Add a footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Created with Streamlit and PandasAI | Your Name/Company
    </div>
    """,
    unsafe_allow_html=True
)
