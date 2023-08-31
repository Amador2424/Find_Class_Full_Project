import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from app import glcm, fct_BiT, haralick_with_mean
from chargement import upload_file

def main():
    st.sidebar.write("### Image Analysis App")

    # Type Image selection
    typeImg = ["CRC", "Glaucoma"]
    r_typeImg = st.sidebar.radio("Select Type Image", typeImg)

    if r_typeImg:
        st.sidebar.write(f"You chose {r_typeImg}")

        is_image_uploaded = upload_file()

        if is_image_uploaded:
            st.write("## Predictions")
            
            query_image = 'uploaded_images/query_image.png'
            img = cv2.imread(query_image)
            B, G, R = cv2.split(img)
            
            R_feat = fct_BiT(R)
            G_feat = glcm(G)
            B_feat = haralick_with_mean(B)
            feature = R_feat + G_feat + B_feat

            # Model selection
            typeModel = ["CatBoost", "LDA", "LightBoost", "SVM"]
            r_typeModel = st.sidebar.radio("Select Model Type", typeModel)

            model_path = f'models/{r_typeImg}/{r_typeModel}.pkl'
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            data_1d = np.array([feature])
            predi = model.predict(data_1d)[0]

            def fct_cls_img():
                if r_typeImg == "CRC":
                    mapping = {
                        "TUM": "01_TUMOR",
                        "STROM": "02_STROMA",
                        "COM": "03_COMPLEX",
                        "LYMP": "04_LYMPHO",
                        "DEB": "05_DEBRIS",
                        "MUC": "06_MUCOSA",
                        "ADIP": "07_ADIPOSE"
                    }
                    return mapping.get(predi, "Unknown")

                elif r_typeImg == "Glaucoma":
                    return "Glaucoma" if predi == 0 else "Saudavel"

            predicted_class = fct_cls_img()
            st.write(f"Predicted Class: {predicted_class}")

    else:
        st.write("Welcome! Please upload an image to get started ...")

if __name__ == "__main__":
    main()
