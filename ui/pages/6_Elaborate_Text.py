import streamlit as st
from ui.Homepage import elaborate_text

st.title("Elaborate")
text_input = st.text_area("Enter Text:", key="elaborate_input")

# Convert button
if st.button("Elaborate"):
    try:
        with st.spinner("Elaborating..."):
            summarized_text = elaborate_text(text_input)
            st.text(summarized_text)
    except Exception as e:
        print(f"Elaboration error: {e}")
        st.error("Elaborate currently not available")