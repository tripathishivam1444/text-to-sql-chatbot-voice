import streamlit as st
import pandas as pd
from utils import create_Xsql_agent, clear_database, upload_to_database, table_exists

st.set_page_config(
    page_title="Travel Database Chatbot",
    page_icon="🧳",
    # layout="wide",
    initial_sidebar_state= "collapsed"
)








if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the agent only if the table exists
if 'agent' not in st.session_state:
    st.session_state.agent = create_Xsql_agent()

with st.sidebar:
    st.title("Data Management")

    # Only show upload if the table does not already exist
    if not table_exists():
        uploaded_file = st.file_uploader("Upload Travel Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head(3), use_container_width=True)

                if st.button("Upload to Database"):
                    with st.spinner("Uploading data to database..."):
                        success, message = upload_to_database(df)
                        if success:
                            st.session_state.agent = create_Xsql_agent()
                            st.success(message)
                        else:
                            st.error(message)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    if st.button("Clear Database"):
        with st.spinner("Clearing database..."):
            success, message = clear_database()
            if success:
                st.session_state.agent = None
                st.session_state.chat_history = []
                st.success(message)
            else:
                st.error(message)

st.title("🧳 Travel Database Chatbot")
st.subheader("Ask about travel destinations")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about travel destinations..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        try:
            with st.spinner("Thinking..."):
                if st.session_state.agent is None:
                    st.error("Agent not initialized. Please upload data first.")
                else:
                    result = st.session_state.agent.invoke(prompt + "LIMIT 15")
                    full_response = result.get('output', 'No response generated')
                    st.markdown(full_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
