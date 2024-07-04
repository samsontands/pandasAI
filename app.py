import streamlit as st
import pandas as pd
from pandasai import PandasAI
import openai

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Example dataset
data = {
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
}
df = pd.DataFrame(data)

# Initialize PandasAI
pandas_ai = PandasAI()

# Streamlit app
st.title("PandasAI with Streamlit")
st.write("Ask questions about your data in natural language!")

# Display the dataframe
st.dataframe(df)

# Input box for user queries
user_query = st.text_input("Enter your query:")

if user_query:
    # Use PandasAI to answer the query
    response = pandas_ai.run(df, user_query)
    st.write("Response:", response)
