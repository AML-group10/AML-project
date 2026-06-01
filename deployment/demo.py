import io

import requests
import streamlit as st
from PIL import Image

st.title("Text to Image Generation")
prompt = st.text_input("What's your prompt?")
generate_button = st.button("Generate image", disabled=not prompt)

if generate_button:
    with st.spinner("Wait while the image is being generated"):
        try:
            response = requests.post(
                "http://localhost:8000/generate", json={"prompt": prompt}
            )
            if response.status_code == 200:
                response.raise_for_status()
                generated_image = Image.open(io.BytesIO(response.content))
                st.image(generated_image)
                st.balloons()
            elif response.status_code == 404:
                st.error("Error 404: Page not found")
            elif response.status_code == 500:
                st.error("Error 500: Internal server error.")
        except requests.exceptions.ConnectionError:
            st.error("Connection to API failed.")
