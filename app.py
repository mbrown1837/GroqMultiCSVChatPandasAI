import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI CSV Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_api_key():
    """Get API key and display source"""
    load_dotenv()
    api_key = os.environ.get('GROQ_API_KEY')
    
    if api_key:
        st.sidebar.success("API Source: Environment File (.env)")
        return api_key
    
    try:
        api_key = st.secrets["groq_api_key"]
        st.sidebar.success("API Source: Streamlit Secrets")
        return api_key
    except:
        st.error("GROQ API key not found!")
        st.stop()

def safe_chat_with_csv(df, query):
    """Enhanced safe chat function with error handling"""
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.7
        )
        
        # Configure PandasAI with enhanced settings
        pandas_ai = SmartDataframe(
            df, 
            config={
                "llm": llm,
                "enable_cache": True,
                "custom_plot": True,
                "save_charts": False,
                "verbose": True,
                "enforce_privacy": True,
                "max_rows": 500000,
                "max_columns": 100
            }
        )
        
        # Enhanced prompt handling
        if "summary" in query.lower():
            return pandas_ai.chat(f"Provide a detailed summary of this dataset including number of rows, columns, and key information")
        elif "analyze" in query.lower():
            return pandas_ai.chat(f"Analyze the data and provide insights about {query}")
        elif "plot" in query.lower() or "graph" in query.lower():
            return pandas_ai.chat(f"Create a visualization for {query}")
        else:
            return pandas_ai.chat(query)

    except Exception as e:
        return f"Error: {str(e)}"

def safe_edit_csv(df, edit_query):
    """Enhanced safe edit function with validation"""
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        pandas_ai = SmartDataframe(
            df, 
            config={
                "llm": llm,
                "enable_cache": True,
                "max_rows": 500000
            }
        )
        
        # Process edit query
        result = pandas_ai.chat(f"Edit the dataframe: {edit_query}")
        
        # Validate result
        if isinstance(result, pd.DataFrame):
            if len(result) > 0:
                return result, "Changes applied successfully!"
            else:
                return df, "Edit resulted in empty DataFrame. Original data retained."
        else:
            return df, f"Edit could not be applied: {str(result)}"
            
    except Exception as e:
        return df, f"Error editing data: {str(e)}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_df' not in st.session_state:
    st.session_state.current_df = None

# Sidebar
with st.sidebar:
    st.title("CSV File Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            st.session_state.current_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")
            
            # Display DataFrame info
            st.write("### Data Info")
            st.info(f"Rows: {st.session_state.current_df.shape[0]}")
            st.info(f"Columns: {st.session_state.current_df.shape[1]}")
            
            # Display column info
            st.write("### Columns")
            for col in st.session_state.current_df.columns:
                st.write(f"- {col}")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.session_state.current_df = None

# Main content
st.title("CSV Chat Assistant")

if st.session_state.current_df is not None:
    try:
        # Safe data preview with error handling
        st.subheader("Data Preview")
        preview_df = st.session_state.current_df.head(3).copy()
        st.dataframe(preview_df)
        
        # Chat and Edit tabs
        tab1, tab2 = st.tabs(["Chat", "Edit"])
        
        with tab1:
            # Chat interface
            query = st.text_input("Chat with your data:", 
                                placeholder="Ask me anything about your data")
            
            if st.button("Send"):
                if query:
                    # Add user message to chat
                    st.session_state.chat_history.append(("user", query))
                    
                    # Get AI response with loading indicator
                    with st.spinner("Processing..."):
                        response = safe_chat_with_csv(st.session_state.current_df, query)
                        st.session_state.chat_history.append(("assistant", response))
            
            # Display chat history
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.write("You:", message)
                else:
                    st.write("Assistant:", message)
                st.write("---")
        
        with tab2:
            # Edit interface
            edit_query = st.text_input("Enter edit instructions:", 
                                     placeholder="Example: Remove duplicates")
            
            if st.button("Apply Edit"):
                if edit_query:
                    with st.spinner("Applying changes..."):
                        edited_df, message = safe_edit_csv(st.session_state.current_df, 
                                                         edit_query)
                        
                        if edited_df is not None:
                            st.session_state.current_df = edited_df
                            st.success(message)
                            st.dataframe(edited_df)
                            
                            # Download button
                            st.download_button(
                                "Download Edited CSV",
                                edited_df.to_csv(index=False),
                                "edited_data.csv",
                                "text/csv"
                            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
else:
    st.info("Please upload a CSV file to begin")

# Helper text
st.sidebar.markdown("""
### Tips:
- Try asking about data summary
- Request specific analysis
- Ask for trends or patterns
- Edit data with clear instructions
""")
