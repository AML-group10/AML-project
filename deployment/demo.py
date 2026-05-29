import io

import requests
import streamlit as st
from PIL import Image

st.title("Text to Image (good for meme generation...)")
prompt = st.text_input("What's your prompt?")
generate_button = st.button("Generate image", disabled=not prompt)

if generate_button:
    try:
        response = requests.post(
            "http://localhost:8000/generate", json={"prompt": prompt}
        )
        print(response)
        print(response.json)
        response.raise_for_status()
        generated_image = Image.open(io.BytesIO(response.content))
        st.image(generated_image)
    except:
        pass
