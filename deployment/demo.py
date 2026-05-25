import requests
import streamlit as st

st.title("Text to Image (good for meme generation...)")
prompt = st.text_input("What's your prompt?")
generate_button = st.button("Generate image", disabled=not prompt)

if generate_button:
    try:
        response = requests.post("http://localhost:8000/", json={"prompt": prompt})
    except:
        pass
