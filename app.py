import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import plotly.express as px
from datetime import datetime

# Loading environment variables from .env file
load_dotenv() 

# Function to chat with and analyze CSV data
def chat_with_csv(df, query):
    groq_api_key = os.environ['GROQ_API_KEY']
    
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama3-70b-8192",
        temperature=0.2
    )
    
    # Initialize SmartDataframe with enhanced capabilities
    pandas_ai = SmartDataframe(
        df, 
        config={
            "llm": llm,
            "save_charts": True,
            "save_charts_path": os.getcwd(),
            "enable_cache": True
        }
    )
    
    try:
        result = pandas_ai.chat(query)
        return result
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Function to edit CSV with AI
def edit_csv_with_ai(df, edit_query):
    try:
        pandas_ai = SmartDataframe(df)
        edited_df = pandas_ai.chat(edit_query)
        return edited_df
    except Exception as e:
        st.error(f"Error editing data: {str(e)}")
        return df

# Function to save changes
def save_changes(df, filename):
    try:
        # Create backup before saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{timestamp}_{filename}"
        df.to_csv(backup_filename, index=False)
        
        # Save main file
        df.to_csv(filename, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

# Function to validate data types
def validate_dataframe(df):
    issues = []
    for column in df.columns:
        null_count = df[column].isnull().sum()
        if null_count > 0:
            issues.append(f"Column '{column}' has {null_count} null values")
    return issues

# Set page configuration
st.set_page_config(
    page_title="Advanced CSV Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Advanced CSV Manager with AI")
st.markdown("""
This application allows you to:
- Analyze CSV data using natural language queries
- Edit data using AI-powered instructions
- Visualize data with automatic charts
- Validate and clean your data
""")

# Sidebar configuration
with st.sidebar:
    st.header("Upload Files")
    input_csvs = st.file_uploader(
        "Upload your CSV files",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if input_csvs:
        st.success(f"📁 {len(input_csvs)} files uploaded")

# Main application logic
if input_csvs:
    # File selection
    selected_file = st.selectbox(
        "Select a CSV file to work with",
        [file.name for file in input_csvs]
    )
    selected_index = [file.name for file in input_csvs].index(selected_file)
    
    # Load and display initial data
    data = pd.read_csv(input_csvs[selected_index])
    st.subheader("Data Preview")
    st.dataframe(data.head(5), use_container_width=True)
    
    # Data information
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Rows: {data.shape[0]}")
    with col2:
        st.info(f"Columns: {data.shape[1]}")
    with col3:
        st.info(f"Memory Usage: {data.memory_usage().sum() / 1024:.2f} KB")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Analysis", "Edit Data", "Data Validation"])
    
    # Analysis Tab
    with tab1:
        st.header("Data Analysis")
        analysis_query = st.text_area(
            "Enter your analysis query",
            placeholder="Example: Calculate the average of column X grouped by column Y"
        )
        
        if analysis_query:
            if st.button("Analyze Data"):
                with st.spinner("Processing analysis..."):
                    result = chat_with_csv(data, analysis_query)
                    st.success("Analysis Complete!")
                    st.write(result)
    
    # Edit Tab
    with tab2:
        st.header("AI-Powered Data Editing")
        edit_query = st.text_area(
            "Enter your editing instructions",
            placeholder="Example: Remove duplicates and standardize column names"
        )
        
        if edit_query:
            if st.button("Edit Data"):
                with st.spinner("Processing edits..."):
                    edited_data = edit_csv_with_ai(data, edit_query)
                    st.success("Edits Complete!")
                    st.dataframe(edited_data, use_container_width=True)
                    
                    if st.button("Save Changes"):
                        if save_changes(edited_data, selected_file):
                            st.success("Changes saved successfully!")
                            st.download_button(
                                "Download Edited CSV",
                                edited_data.to_csv(index=False),
                                file_name=f"edited_{selected_file}",
                                mime="text/csv"
                            )
    
    # Validation Tab
    with tab3:
        st.header("Data Validation")
        if st.button("Run Validation"):
            with st.spinner("Validating data..."):
                issues = validate_dataframe(data)
                if issues:
                    st.warning("Validation Issues Found:")
                    for issue in issues:
                        st.write(f"- {issue}")
                else:
                    st.success("No validation issues found!")
                
                # Display data types
                st.subheader("Column Data Types")
                st.write(data.dtypes)

else:
    st.info("👆 Please upload your CSV files using the sidebar")
    
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit, PandasAI, and Groq LLM</p>
    </div>
    """,
    unsafe_allow_html=True
)
