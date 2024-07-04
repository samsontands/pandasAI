import os
import pandas as pd
import streamlit as st
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse

# Create sample dataframes
employees_df = pd.DataFrame({
    "EmployeeID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Olivia", "William"],
    "Department": ["HR", "Sales", "IT", "Marketing", "Finance"],
})

salaries_df = pd.DataFrame({
    "EmployeeID": [1, 2, 3, 4, 5],
    "Salary": [5000, 6000, 4500, 7000, 5500],
})

# Set the PandasAI API key (make sure to set this in your Streamlit secrets)
os.environ["PANDASAI_API_KEY"] = st.secrets["PANDASAI_API_KEY"]

# Create the PandasAI agent
agent = Agent(
    [employees_df, salaries_df],
    config={"verbose": True, "response_parser": StreamlitResponse},
)

# Use Streamlit to display the data and results
st.title("PandasAI with Streamlit")
st.write("Employee Data")
st.write(employees_df)
st.write("Salaries Data")
st.write(salaries_df)

# Generate and display the chart
agent.chat("Plot salaries against employee name")
