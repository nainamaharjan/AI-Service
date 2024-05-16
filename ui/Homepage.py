import os
import time
import requests
import tempfile
import streamlit as st


BASE_URL = "http://0.0.0.0:8000"


def shorten_text(text):
    api_url = f"{BASE_URL}/shorten-text"
    params = {"text": text}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        llm_response = response.json().get("response")
        return llm_response
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return None


def correct_grammar(text):
    api_url = f"{BASE_URL}/check-grammar"
    params = {"text": text}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        llm_response = response.json().get("response")
        return llm_response
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return None


def make_professional(text):
    api_url = f"{BASE_URL}//convert-to-professional"
    params = {"text": text}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        llm_response = response.json().get("response")
        return llm_response
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return None


def make_casual(text):
    api_url = f"{BASE_URL}/convert-to-casual"
    params = {"text": text}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        llm_response = response.json().get("response")
        return llm_response
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return None

def elaborate_text(text):
    api_url = f"{BASE_URL}/elaborate-text"
    params = {"text": text}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        llm_response = response.json().get("response")
        return llm_response
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return None

def ask_llm(text):
    api_url = f"{BASE_URL}/response-from-llm"
    params = {"text": text}
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        llm_response = response.json().get("response")
        return llm_response
    except requests.RequestException as e:
        st.error(f"Error: {e}")
        return None

def enter_prompt_and_get_response(prompt):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            user_input = st.session_state.messages[-1]["content"]
            response = ask_llm(text=user_input)

            st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
    except Exception as e:
        print(f"LLM response generation error: {e}")
        st.error("Assistant currently not available")
