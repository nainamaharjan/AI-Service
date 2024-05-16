import streamlit as st
from ui.Homepage import correct_grammar


st.title("Correct Grammar")
text_input = st.text_area("Enter Text:", key="tts_input")

# Convert button
if st.button("Correct Grammar"):
    try:
        with st.spinner("Checking grammar..."):
            correct_grammar = correct_grammar(text_input)
            st.text(correct_grammar)
    except Exception as e:
        print(f"Grammar check error: {e}")
        st.error("Grammar Checker currently not available")
