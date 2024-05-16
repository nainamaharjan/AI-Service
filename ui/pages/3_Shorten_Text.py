import streamlit as st
from ui.Homepage import shorten_text

st.title("Shorten Text")
text_input = st.text_area("Enter Text:", key="shorten_input")

# Convert button
if st.button("Shorten"):
    try:
        with st.spinner("Summarizing..."):
            summarized_text = shorten_text(text_input)
            st.text(summarized_text)
    except Exception as e:
        print(f"Summarization error: {e}")
        st.error("Summarizer currently not available")
