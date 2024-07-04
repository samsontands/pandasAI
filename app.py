import os
import pandas as pd
import streamlit as st
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse

# Set the PandasAI API key (make sure to set this in your Streamlit secrets)
os.environ["PANDASAI_API_KEY"] = st.secrets["PANDASAI_API_KEY"]

# Streamlit app layout
st.title("PandasAI with Streamlit")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame Preview:", df.head())

    # Create the PandasAI agent with the uploaded DataFrame
    agent = Agent([df], config={"verbose": True, "response_parser": StreamlitResponse})

    # Input box for user queries
    user_query = st.text_input("Ask a question about your data:")

    # Generate and display the response
    if user_query:
        response = agent.chat(user_query)
        st.write("Response:")
        if isinstance(response, list):
            for res in response:
                if isinstance(res, dict) and "chart" in res:
                    st.pyplot(res["chart"])
                else:
                    st.write(res)
        else:
            st.write(response)
