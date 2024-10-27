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
        background-color: #2e7bf6;
        color: white;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #2e7bf6;
        color: white;
        margin-left: 20%;
    }
    .ai-message {
        background-color: #f0f2f6;
        margin-right: 20%;
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
                "custom_plot": True
            }
        )
        
        if is_edit:
            result = pandas_ai.chat(f"Edit the dataframe as follows: {query}")
            return (result, "✅ Changes applied successfully!") if isinstance(result, pd.DataFrame) else (df, str(result))
        else:
            return pandas_ai.chat(query)

    except Exception as e:
        return f"❌ Error: {str(e)}"

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
        st.success(f"📊 {uploaded_file.name} loaded successfully!")
        
        # Quick stats
        st.write("### Data Info")
        st.info(f"📏 Rows: {st.session_state.current_df.shape[0]}")
        st.info(f"📊 Columns: {st.session_state.current_df.shape[1]}")

# Main content
st.title("🤖 AI CSV Assistant")

if st.session_state.current_df is not None:
    # Compact data preview
    st.write("### Data Preview")
    st.dataframe(st.session_state.current_df.head(3), use_container_width=True, height=150)
    
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["💬 Chat", "✏️ Edit"])
    
    with tab1:
        # Chat interface
        query = st.text_input("Ask anything about your data:", 
                            key="chat_input",
                            placeholder="Example: Give me a summary of this dataset")
        
        if st.button("Send", key="send_chat"):
            st.session_state.chat_history.append(("user", query))
            with st.spinner("🤔 Thinking..."):
                response = process_csv_query(st.session_state.current_df, query)
                st.session_state.chat_history.append(("ai", response))
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            display_chat_message(message, is_user=(role == "user"))
    
    with tab2:
        # Edit interface
        edit_query = st.text_input("Enter edit instructions:", 
                                key="edit_input",
                                placeholder="Example: Remove duplicate rows")
        
        if st.button("Edit", key="send_edit"):
            with st.spinner("✏️ Applying changes..."):
                edited_df, message = process_csv_query(st.session_state.current_df, 
                                                    edit_query, 
                                                    is_edit=True)
                st.success(message)
                st.session_state.current_df = edited_df
                st.dataframe(edited_df, use_container_width=True)
                
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
