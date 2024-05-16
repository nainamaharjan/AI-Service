import streamlit as st
from ui.Homepage import enter_prompt_and_get_response


st.title("Nepali AI assistant demo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
prompt = st.chat_input("Enter your input")
if prompt:
    enter_prompt_and_get_response(prompt)
