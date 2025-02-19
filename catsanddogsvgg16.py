import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import requests
import os

# URL do modelo no GitHub (altere para o link correto do raw file)
MODEL_URL = "https://github.com/andrepporto/IFRO-DataScience/blob/main/modelo_cachorro_gato.h5"
MODEL_PATH = "modelo_cachorro_gato.h5"

# Baixar o modelo se ele não existir
if not os.path.exists(MODEL_PATH):
    with st.spinner("Baixando modelo... Isso pode levar alguns segundos."):
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Carregar o modelo treinado
model = load_model(MODEL_PATH)

# Interface do Usuário
st.title("Classificador de Gatos e Cachorros 🐶🐱")
uploaded_file = st.file_uploader("Envie uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", use_column_width=True)

    # Preprocessamento
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predição
    prediction = model.predict(img_array)
    label = "Cachorro 🐶" if prediction[0][0] >= 0.5 else "Gato 🐱"

    st.write(f"### Resultado: {label}")
