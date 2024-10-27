import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Set page configuration
st.set_page_config(
    page_title="AI CSV Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for chat interface
st.markdown("""
    <style>
    .chat-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #2E7BF6;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #383838;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stButton > button {
        background-color: #2E7BF6;
        color: white;
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def get_api_key():
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
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama-3.1-70b-versatile",
            temperature=0.7
        )
        
        pandas_ai = SmartDataframe(
            df.copy(),
            config={
                "llm": llm,
                "enable_cache": True,
                "custom_plot": True,
                "verbose": True
            }
        )
        
        response = pandas_ai.chat(query)
        return response

    except Exception as e:
        return f"Error: {str(e)}"

def edit_csv(df, edit_query):
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama-3.1-70b-versatile",
            temperature=0.2
        )
        
        pandas_ai = SmartDataframe(df.copy(), config={"llm": llm})
        result = pandas_ai.chat(f"Edit the dataframe: {edit_query}")
        
        if isinstance(result, pd.DataFrame):
            return result, "Changes applied successfully!"
        return df, str(result)
            
    except Exception as e:
        return df, f"Error: {str(e)}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_df' not in st.session_state:
    st.session_state.current_df = None

# Sidebar
with st.sidebar:
    st.title("📁 File Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            st.session_state.current_df = pd.read_csv(uploaded_file)
            st.success(f"✅ {uploaded_file.name} loaded")
            st.write("### Data Info")
            st.info(f"📊 Rows: {st.session_state.current_df.shape[0]}")
            st.info(f"📊 Columns: {st.session_state.current_df.shape[1]}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.session_state.current_df = None

# Main content
st.title("💬 CSV Chat Assistant")

if st.session_state.current_df is not None:
    # Data preview
    with st.expander("Preview Data"):
        st.dataframe(st.session_state.current_df.head(3), use_container_width=True)
    
    # Tabs for chat and edit
    tab1, tab2 = st.tabs(["💭 Chat", "✏️ Edit"])
    
    with tab1:
        query = st.text_input("Ask anything about your data:", 
                            placeholder="Example: Give me a summary of this dataset")
        
        col1, col2 = st.columns([6,1])
        with col2:
            send_button = st.button("Send 📤")
        
        if send_button and query:
            st.session_state.chat_history.append(("user", query))
            with st.spinner("Thinking..."):
                response = chat_with_csv(st.session_state.current_df, query)
                st.session_state.chat_history.append(("assistant", response))
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="user-message">👤 {message}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 {message}</div>', 
                          unsafe_allow_html=True)
    
    with tab2:
        edit_query = st.text_input("Enter edit instructions:", 
                                 placeholder="Example: Remove duplicate rows")
        
        col1, col2 = st.columns([6,1])
        with col2:
            edit_button = st.button("Edit ✏️")
        
        if edit_button and edit_query:
            with st.spinner("Applying changes..."):
                edited_df, message = edit_csv(st.session_state.current_df, edit_query)
                if isinstance(edited_df, pd.DataFrame):
                    st.session_state.current_df = edited_df
                    st.success(message)
                    st.dataframe(edited_df)
                    
                    st.download_button(
                        "📥 Download Edited CSV",
                        edited_df.to_csv(index=False),
                        "edited_data.csv",
                        "text/csv"
                    )
                else:
                    st.error(message)

else:
    st.info("👈 Please upload a CSV file to begin")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        Built with Streamlit, PandasAI, and Groq LLM
    </div>
    """, 
    unsafe_allow_html=True
)
