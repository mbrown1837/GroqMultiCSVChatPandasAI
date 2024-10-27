import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import plotly.express as px
from datetime import datetime

def get_api_key():
    """Get API key from either .env file or Streamlit secrets"""
    # Try getting API key from .env first
    load_dotenv()
    api_key = os.environ.get('GROQ_API_KEY')
    
    # If not in .env, try getting from Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["groq_api_key"]
        except:
            st.error("GROQ API key not found in .env file or Streamlit secrets")
            st.stop()
    
    return api_key

def chat_with_csv(df, query):
    try:
        groq_api_key = get_api_key()
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        pandas_ai = SmartDataframe(
            df, 
            config={
                "llm": llm,
                "save_charts": True,
                "save_charts_path": os.getcwd(),
                "enable_cache": True
            }
        )
        
        result = pandas_ai.chat(query)
        
        # Handle different types of results
        if isinstance(result, str):
            st.write(result)
        elif isinstance(result, pd.DataFrame):
            st.dataframe(result)
        else:
            st.write(result)
            
        return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def edit_csv_with_ai(df, edit_query):
    try:
        groq_api_key = get_api_key()
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        edited_df = pandas_ai.chat(edit_query)
        if isinstance(edited_df, pd.DataFrame):
            return edited_df
        else:
            st.warning("AI response was not a DataFrame. Returning original data.")
            return df
    except Exception as e:
        st.error(f"Error editing data: {str(e)}")
        return df

def save_changes(df, filename):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{timestamp}_{filename}"
        df.to_csv(backup_filename, index=False)
        
        output_filename = f"edited_{filename}"
        df.to_csv(output_filename, index=False)
        return True, output_filename
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False, None

# Set page configuration
st.set_page_config(
    page_title="Advanced CSV Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verify API key at startup
api_key = get_api_key()
if not api_key:
    st.stop()

# Application title and description
st.title("Advanced CSV Manager with AI")
st.markdown("""
This application allows you to:
- Analyze CSV data using natural language queries
- Edit data using AI-powered instructions
- Visualize data with automatic charts
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
    try:
        selected_file = st.selectbox(
            "Select a CSV file to work with",
            [file.name for file in input_csvs]
        )
        
        if selected_file:
            selected_index = [file.name for file in input_csvs].index(selected_file)
            
            try:
                data = pd.read_csv(input_csvs[selected_index])
                if data.empty:
                    st.warning("The selected CSV file is empty")
                else:
                    st.subheader("Data Preview")
                    st.dataframe(data.head(5), use_container_width=True)
                    
                    # Create tabs for different functionalities
                    tab1, tab2 = st.tabs(["Analysis", "Edit Data"])
                    
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
                                    if result is not None:
                                        st.success("Analysis Complete!")
                    
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
                                    if edited_data is not None:
                                        st.success("Edits Complete!")
                                        st.dataframe(edited_data, use_container_width=True)
                                        
                                        if st.button("Save Changes"):
                                            success, saved_filename = save_changes(edited_data, selected_file)
                                            if success:
                                                st.success(f"Changes saved to {saved_filename}")
                                                st.download_button(
                                                    "Download Edited CSV",
                                                    edited_data.to_csv(index=False),
                                                    file_name=saved_filename,
                                                    mime="text/csv"
                                                )
                                    
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                
    except Exception as e:
        st.error(f"Error processing file selection: {str(e)}")
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
