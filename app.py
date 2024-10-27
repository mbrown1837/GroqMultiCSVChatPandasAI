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
    .send-button {
        margin-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

def get_api_key():
    load_dotenv()
    api_key = os.environ.get('GROQ_API_KEY')
    return api_key if api_key else st.secrets["groq_api_key"]

def process_csv_query(df, query, is_edit=False):
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        # Enhanced prompt for better summary and analysis
        if "summary" in query.lower():
            enhanced_query = f"""Provide a comprehensive summary of this dataset including:
            1. What kind of data it contains
            2. Key patterns or trends
            3. Important insights
            Original query: {query}"""
            query = enhanced_query
        
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
    if 'last_query' not in st.session_state:
        st.session_state.last_query = None

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
        st.write("### Quick Stats")
        st.info(f"📏 Rows: {st.session_state.current_df.shape[0]}")
        st.info(f"📊 Columns: {st.session_state.current_df.shape[1]}")

# Main content
st.title("🤖 AI CSV Assistant")

if st.session_state.current_df is not None:
    # Data preview with toggle
    if st.checkbox("Show Data Preview", value=True):
        st.write("### Data Preview")
        st.dataframe(st.session_state.current_df.head(3), use_container_width=True)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "✏️ Edit", "📊 Quick Analysis"])
    
    with tab1:
        # Chat interface with automatic sending
        query = st.text_input("Ask anything about your data:", 
                            key="chat_input",
                            placeholder="Example: Give me a summary of this dataset",
                            on_change=lambda: setattr(st.session_state, 'last_query', st.session_state.chat_input))
        
        # Process query when Enter is pressed
        if query and query != st.session_state.last_query:
            st.session_state.chat_history.append(("user", query))
            with st.spinner("🤔 Thinking..."):
                response = process_csv_query(st.session_state.current_df, query)
                st.session_state.chat_history.append(("ai", response))
            st.session_state.last_query = query
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            display_chat_message(message, is_user=(role == "user"))
    
    with tab2:
        # Edit interface with automatic sending
        edit_query = st.text_input("Enter edit instructions:", 
                                key="edit_input",
                                placeholder="Example: Remove duplicate rows",
                                on_change=lambda: setattr(st.session_state, 'last_edit_query', st.session_state.edit_input))
        
        if edit_query and edit_query != getattr(st.session_state, 'last_edit_query', None):
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
    
    with tab3:
        st.write("### Quick Analysis Tools")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate Summary Statistics"):
                st.write(st.session_state.current_df.describe())
        
        with col2:
            if st.button("🔍 Check Data Quality"):
                missing_values = st.session_state.current_df.isnull().sum()
                st.write("Missing Values per Column:", missing_values)

else:
    st.info("👈 Please upload a CSV file to begin")

# Footer with helpful tips
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        💡 Tips:
        • Press Enter to send messages
        • Use clear, specific questions
        • Try asking for trends and patterns
    </div>
""", unsafe_allow_html=True)
