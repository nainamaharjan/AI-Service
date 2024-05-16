import streamlit as st
from ui.Homepage import make_casual

st.title("Make Text Casual")
text_input = st.text_area("Enter Text:", key="casual_input")

# Convert button
if st.button("Casual"):
    try:
        with st.spinner("Working..."):
            casual_text = make_casual(text_input)
            st.text(casual_text)
    except Exception as e:
        print(f"System error: {e}")
        st.error("Service currently not available")
