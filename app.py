import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Set page configuration with custom theme
st.set_page_config(
    page_title="AI CSV Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .stButton > button {
        border-radius: 20px;
        margin-top: 10px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .ai-message {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

def get_api_key():
    load_dotenv()
    api_key = os.environ.get('GROQ_API_KEY')
    
    if api_key:
        st.sidebar.success("✅ Using API key from .env file")
        return api_key
    
    try:
        api_key = st.secrets["groq_api_key"]
        st.sidebar.success("✅ Using API key from Streamlit secrets")
        return api_key
    except:
        st.error("❌ GROQ API key not found!")
        st.stop()

def process_csv_query(df, query, is_edit=False):
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
                "enable_cache": True,
                "custom_plot": True
            }
        )
        
        if is_edit:
            # For editing queries, try to return DataFrame
            result = pandas_ai.chat(f"Edit the dataframe: {query}")
            if isinstance(result, pd.DataFrame):
                return result, "DataFrame edited successfully!"
            return df, str(result)
        else:
            # For analysis queries
            result = pandas_ai.chat(query)
            return result

    except Exception as e:
        return f"Error: {str(e)}"

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None

def display_chat_message(message, is_user=True):
    message_type = "user-message" if is_user else "ai-message"
    icon = "👤" if is_user else "🤖"
    st.markdown(f"""
        <div class="chat-message {message_type}">
            {icon} {message}
        </div>
    """, unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Sidebar for file upload
with st.sidebar:
    st.title("📁 File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        st.session_state.current_df = pd.read_csv(uploaded_file)
        st.success(f"📊 Loaded: {uploaded_file.name}")
        
        # Display data info
        st.write("### Data Summary")
        st.info(f"Rows: {st.session_state.current_df.shape[0]}")
        st.info(f"Columns: {st.session_state.current_df.shape[1]}")

# Main content
st.title("🤖 AI CSV Assistant")

if st.session_state.current_df is not None:
    # Display data preview
    st.write("### Data Preview")
    st.dataframe(st.session_state.current_df.head(3), use_container_width=True)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["💬 Chat", "✏️ Edit"])
    
    with tab1:
        # Chat interface
        query = st.text_input("Ask anything about your data:", 
                            placeholder="Example: Summarize this dataset",
                            key="chat_input")
        
        col1, col2 = st.columns([6,1])
        with col2:
            send_button = st.button("📤 Send", use_container_width=True)
        
        if query and (send_button or query.endswith('\n')):
            # Add user message to chat history
            st.session_state.chat_history.append(("user", query))
            
            # Get AI response
            response = process_csv_query(st.session_state.current_df, query)
            st.session_state.chat_history.append(("ai", response))
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            display_chat_message(message, is_user=(role == "user"))
    
    with tab2:
        # Edit interface
        edit_query = st.text_input("Enter edit instructions:", 
                                placeholder="Example: Remove duplicate rows",
                                key="edit_input")
        
        col1, col2 = st.columns([6,1])
        with col2:
            edit_button = st.button("✏️ Edit", use_container_width=True)
        
        if edit_query and (edit_button or edit_query.endswith('\n')):
            with st.spinner("Applying changes..."):
                edited_df, message = process_csv_query(st.session_state.current_df, 
                                                    edit_query, 
                                                    is_edit=True)
                st.success(message)
                st.session_state.current_df = edited_df
                st.dataframe(edited_df, use_container_width=True)
                
                # Download button for edited data
                st.download_button(
                    label="📥 Download Edited CSV",
                    data=edited_df.to_csv(index=False),
                    file_name="edited_data.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Please upload a CSV file to begin")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        Made with ❤️ using Streamlit, PandasAI, and Groq LLM
    </div>
    """, 
    unsafe_allow_html=True
)
