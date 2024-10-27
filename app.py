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

def get_api_key():
    load_dotenv()
    api_key = os.environ.get('GROQ_API_KEY')
    
    if api_key:
        st.sidebar.success("🔑 API Source: Environment File (.env)")
        return api_key
    
    try:
        api_key = st.secrets["groq_api_key"]
        st.sidebar.success("🔑 API Source: Streamlit Secrets")
        return api_key
    except:
        st.error("❌ GROQ API key not found!")
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
        
        return pandas_ai.chat(query)

    except Exception as e:
        return f"❌ Error: {str(e)}"

def edit_csv(df, edit_query):
    try:
        llm = ChatGroq(
            groq_api_key=get_api_key(), 
            model_name="llama-3.1-70b-versatile",
            temperature=0.2
        )
        
        df_copy = df.copy()
        pandas_ai = SmartDataframe(
            df_copy,
            config={
                "llm": llm,
                "enable_cache": True,
                "custom_plot": True,
                "verbose": True,
                "enforce_privacy": True,
                "max_rows": None,
                "max_columns": None
            }
        )
        
        result = pandas_ai.chat(
            f"""
            Perform the following edit on the dataframe: {edit_query}
            Important: Maintain data integrity and row count ({len(df)} rows).
            """
        )
        
        if isinstance(result, pd.DataFrame):
            if len(result) == len(df):
                return result, "✅ Changes applied successfully!"
            else:
                return df, f"⚠️ Row count mismatch. Operation cancelled."
        else:
            return df, f"❌ Could not apply edit: {str(result)}"
            
    except Exception as e:
        return df, f"❌ Error: {str(e)}"

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
            
            # Data Information
            st.write("📊 Data Info")
            st.info(f"📈 Rows: {st.session_state.current_df.shape[0]}")
            st.info(f"📉 Columns: {st.session_state.current_df.shape[1]}")
            
            # Column Information
            st.write("📋 Columns List")
            for col in st.session_state.current_df.columns:
                st.write(f"📌 {col}")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.current_df = None

# Main content
st.title("🤖 CSV Assistant")

if st.session_state.current_df is not None:
    # Data preview in expander
    with st.expander("👀 Preview Data"):
        st.dataframe(st.session_state.current_df.head(3), use_container_width=True)
    
    # Chat and Edit tabs
    tab1, tab2 = st.tabs(["💬 Chat", "✏️ Edit"])
    
    with tab1:
        # Chat interface
        query = st.text_input("💭 Ask anything about your data:", 
                            placeholder="Example: Give me a summary of this dataset")
        
        if st.button("📤 Send"):
            if query:
                st.session_state.chat_history.append(("user", query))
                with st.spinner("🤔 Thinking..."):
                    response = chat_with_csv(st.session_state.current_df, query)
                    st.session_state.chat_history.append(("assistant", response))
        
        # Chat history display
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                    <div style='background-color: #2E7BF6; color: white; padding: 10px; 
                    border-radius: 10px; margin: 5px 0; text-align: right;'>
                    👤 {message}</div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #383838; padding: 10px; 
                    border-radius: 10px; margin: 5px 0;'>
                    🤖 {message}</div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        # Edit interface
        edit_query = st.text_input("✏️ Enter edit instructions:", 
                                 placeholder="Example: Remove duplicate rows")
        
        if st.button("🔄 Apply Edit"):
            if edit_query:
                with st.spinner("⚙️ Processing changes..."):
                    edited_df, message = edit_csv(st.session_state.current_df, edit_query)
                    
                    if isinstance(edited_df, pd.DataFrame):
                        st.success(message)
                        st.write("👀 Preview of changes:")
                        st.dataframe(edited_df.head())
                        
                        if st.button("✅ Confirm Changes"):
                            st.session_state.current_df = edited_df
                            st.success("✨ Changes saved!")
                            
                            # Download option
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
        🚀 Built with Streamlit, PandasAI, and Groq LLM
    </div>
    """, 
    unsafe_allow_html=True
)
