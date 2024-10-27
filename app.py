import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Set page configuration with dark theme
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

def chat_with_csv(df, query):
    """Enhanced chat function for more conversational responses"""
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama3-70b-8192",
            temperature=0.7  # Increased for more conversational responses
        )
        
        # Enhanced prompt for more conversational responses
        conversation_prompt = f"""
        Act as a helpful assistant chatting about this CSV data. 
        Be conversational and friendly in your responses.
        Query: {query}
        """
        
        pandas_ai = SmartDataframe(
            df, 
            config={
                "llm": llm,
                "enable_cache": True,
                "custom_plot": True
            }
        )
        
        return pandas_ai.chat(conversation_prompt)

    except Exception as e:
        return f"Error: {str(e)}"

def edit_csv(df, edit_query):
    """Function to edit CSV data"""
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        result = pandas_ai.chat(f"Edit the dataframe: {edit_query}")
        return result if isinstance(result, pd.DataFrame) else df
        
    except Exception as e:
        st.error(f"Error editing data: {str(e)}")
        return df

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
        st.session_state.current_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")
        st.info(f"Rows: {st.session_state.current_df.shape[0]}")
        st.info(f"Columns: {st.session_state.current_df.shape[1]}")

# Main content
st.title("CSV Chat Assistant")

if st.session_state.current_df is not None:
    # Simple data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.current_df.head(3))
    
    # Chat and Edit tabs
    tab1, tab2 = st.tabs(["Chat", "Edit"])
    
    with tab1:
        # Chat interface
        query = st.text_input("Chat with your data:", placeholder="Ask me anything about your data")
        
        if st.button("Send"):
            if query:
                # Add user message to chat
                st.session_state.chat_history.append(("user", query))
                
                # Get AI response
                with st.spinner("Processing..."):
                    response = chat_with_csv(st.session_state.current_df, query)
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
        edit_query = st.text_input("Enter edit instructions:", placeholder="Example: Remove duplicates")
        
        if st.button("Apply Edit"):
            if edit_query:
                with st.spinner("Applying changes..."):
                    edited_df = edit_csv(st.session_state.current_df, edit_query)
                    st.session_state.current_df = edited_df
                    st.success("Changes applied!")
                    st.dataframe(edited_df)
                    
                    # Download button
                    st.download_button(
                        "Download Edited CSV",
                        edited_df.to_csv(index=False),
                        "edited_data.csv",
                        "text/csv"
                    )

else:
    st.info("Please upload a CSV file to begin")
