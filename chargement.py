import streamlit as st
from PIL import Image
import os

def upload_file():
    # Upload image using Streamlit
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "PNM", "tiff"])
    
    if uploaded_file is not None:
        # Folder to store the uploaded image
        folder = "uploaded_images"
        # Create the folder if not exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Specify a new filename
        new_file_name = "query_image.png"
        
        # Construct the file path with the new file name
        file_path = os.path.join(folder, new_file_name)
        
        # Write uploaded image with the new file name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Read the uploaded image
        image = Image.open(file_path)
        
        # Display the upload image with Streamlit
        st.image(image, caption="Query Image")
        return True
    else:
        st.write("No file uploaded")
        return False
            