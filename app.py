import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

def get_api_key():
    """Get API key from either .env file or Streamlit secrets"""
    # Try getting API key from .env first
    load_dotenv()
    api_key = os.environ.get('GROQ_API_KEY')
    
    if api_key:
        st.sidebar.success("Using API key from .env file")
        return api_key
    
    # If not in .env, try getting from Streamlit secrets
    try:
        api_key = st.secrets["groq_api_key"]
        st.sidebar.success("Using API key from Streamlit secrets")
        return api_key
    except:
        st.error("GROQ API key not found in .env file or Streamlit secrets")
        st.stop()

def chat_with_csv(df, query):
    try:
        groq_api_key = get_api_key()
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        # Modified config to fix PandasAI error
        pandas_ai = SmartDataframe(
            df, 
            config={
                "llm": llm,
                "enable_cache": True,
                "custom_plot": True  # Use this instead of save_charts
            }
        )
        
        result = pandas_ai.chat(query)
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
        return edited_df if isinstance(edited_df, pd.DataFrame) else df
    except Exception as e:
        st.error(f"Error editing data: {str(e)}")
        return df

# Set page configuration
st.set_page_config(
    page_title="CSV Manager",
    page_icon="📊",
    layout="wide"
)

# Verify API key at startup
api_key = get_api_key()

# Application title
st.title("CSV Manager with AI")

# Sidebar for file upload
with st.sidebar:
    input_csvs = st.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True
    )

# Main application logic
if input_csvs:
    selected_file = st.selectbox(
        "Select CSV file",
        [file.name for file in input_csvs]
    )
    
    if selected_file:
        selected_index = [file.name for file in input_csvs].index(selected_file)
        data = pd.read_csv(input_csvs[selected_index])
        
        st.dataframe(data.head(5), use_container_width=True)
        
        tab1, tab2 = st.tabs(["Analysis", "Edit Data"])
        
        # Analysis Tab
        with tab1:
            # Always show the button
            st.button("Analyze", key="analyze_button", type="primary")
            
            # Text input that triggers on Enter
            query = st.text_area(
                "Enter your query",
                key="analysis_query",
                height=100,
                placeholder="Example: Calculate the average of column X",
                on_change=lambda: st.session_state.update({'analyze_clicked': True}) 
                if 'analysis_query' in st.session_state and st.session_state.analysis_query 
                else None
            )
            
            # Process query on Enter or button click
            if (query and st.session_state.get('analyze_clicked', False)) or st.session_state.get('analyze_button', False):
                with st.spinner("Processing..."):
                    result = chat_with_csv(data, query)
                    if result is not None:
                        st.success("Analysis Complete!")
                        st.write(result)
                st.session_state['analyze_clicked'] = False
        
        # Edit Tab
        with tab2:
            # Always show the button
            st.button("Edit", key="edit_button", type="primary")
            
            # Text input that triggers on Enter
            edit_query = st.text_area(
                "Enter edit instructions",
                key="edit_query",
                height=100,
                placeholder="Example: Remove duplicates",
                on_change=lambda: st.session_state.update({'edit_clicked': True}) 
                if 'edit_query' in st.session_state and st.session_state.edit_query 
                else None
            )
            
            # Process edit on Enter or button click
            if (edit_query and st.session_state.get('edit_clicked', False)) or st.session_state.get('edit_button', False):
                with st.spinner("Processing..."):
                    edited_data = edit_csv_with_ai(data, edit_query)
                    st.success("Edits Complete!")
                    st.dataframe(edited_data)
                    
                    if st.download_button(
                        "Download Edited CSV",
                        edited_data.to_csv(index=False),
                        file_name=f"edited_{selected_file}",
                        mime="text/csv"
                    ):
                        st.success("File downloaded successfully!")
                st.session_state['edit_clicked'] = False

else:
    st.info("Please upload a CSV file to begin")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Built with Streamlit, PandasAI, and Groq LLM</div>", 
    unsafe_allow_html=True
)
