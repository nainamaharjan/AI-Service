import streamlit as st
from ui.Homepage import make_professional

st.title("Make Text Professional")
text_input = st.text_area("Enter Text:", key="prof_input")

# Convert button
if st.button("Professional"):
    try:
        with st.spinner("Working..."):
            professional_text = make_professional(text_input)
            st.text(professional_text)
    except Exception as e:
        print(f"System error: {e}")
        st.error("Service currently not available")
