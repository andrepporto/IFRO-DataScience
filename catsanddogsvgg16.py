import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Carregar modelo treinado
model = load_model("modelo_cachorro_gato.h5")

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
