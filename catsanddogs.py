#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/andrepporto/IFRO-DataScience/blob/main/catsanddogs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import numpy as np
import os
import keras
import tensorflow as tf
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

get_ipython().system('curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip')
get_ipython().system('unzip -q kagglecatsanddogs_5340.zip')
get_ipython().system('ls')


# In[ ]:


num_skipped = 0
for folder_name in ("Cat", "Dog"):
  folder_path = os.path.join("PetImages", folder_name)
  for fname in os.listdir(folder_path):
    fpath = os.path.join(folder_path, fname)
    try:
      fobj = open(fpath, "rb")
      is_jfif = b"JFIF" in fobj.peek(10)
    finally:
      fobj.close()

    if not is_jfif:
      num_skipped += 1
      os.remove(fpath)

print(f"Deleted {num_skipped} images.")


# In[ ]:


image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split = 0.2,
    subset = "both",
    seed = 1337,
    image_size = image_size,
    batch_size =batch_size,
)


# In[ ]:


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# In[ ]:


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


# In[ ]:


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")


# In[ ]:


train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


# In[ ]:


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(size, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(size, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

      residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
      x = layers.add([x, residual])
      previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
      units = 1
    else:
      units = num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


epochs = 13

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)


# In[ ]:


img = keras.utils.load_img("PetImages/Cat/6769.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


# In[ ]:


get_ipython().system('pip install streamlit')
get_ipython().system('pip install pyngrok')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport tensorflow as tf\nfrom tensorflow.keras.models import load_model\n\n# Função para salvar um modelo\ndef save_model(model, filepath="my_model.h5"):\n    model.save(filepath)\n\n# Função para carregar um modelo salvo\ndef load_uploaded_model(uploaded_file):\n    if uploaded_file is not None:\n        with open("uploaded_model.h5", "wb") as f:\n            f.write(uploaded_file.getbuffer())\n        return load_model("uploaded_model.h5")\n    return None\n\nst.title("Save and Load TensorFlow Model with Streamlit")\n\nst.header("Save Model")\nif st.button("Save Model"):\n    model = tf.keras.Sequential([\n        tf.keras.layers.Dense(10, activation="relu", input_shape=(20,)),\n        tf.keras.layers.Dense(1, activation="sigmoid")\n    ])\n    save_model(model, "my_model.h5")\n    st.success("Model saved as my_model.h5!")\n\nst.header("Load Uploaded Model")\nuploaded_file = st.file_uploader("Upload a TensorFlow Model (.h5 file)", type=["h5"])\nif uploaded_file:\n    loaded_model = load_uploaded_model(uploaded_file)\n    if loaded_model:\n        st.success("Model loaded successfully!")\n        st.write(loaded_model.summary())\n    else:\n        st.error("Failed to load the model!")\n')


# In[ ]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Função para salvar um modelo
def save_model(model, filepath="my_model.h5"):
    model.save(filepath)

# Função para carregar um modelo salvo
def load_uploaded_model(uploaded_file):
    if uploaded_file is not None:
        # Salva o arquivo temporariamente
        with open("uploaded_model.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Carrega o modelo salvo
        return load_model("uploaded_model.h5")
    return None

# Streamlit Interface
st.title("Save and Load TensorFlow Model with Streamlit")

# Parte 1: Salvar o modelo
st.header("Save Model")
if st.button("Save Model"):
    # Exemplo de modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(20,)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    save_model(model, "my_model.h5")
    st.success("Model saved as my_model.h5!")

# Parte 2: Fazer upload e carregar modelo
st.header("Load Uploaded Model")
uploaded_file = st.file_uploader("Upload a TensorFlow Model (.h5 file)", type=["h5"])
if uploaded_file:
    loaded_model = load_uploaded_model(uploaded_file)
    if loaded_model:
        st.success("Model loaded successfully!")
        st.write(loaded_model.summary())
    else:
        st.error("Failed to load the model!")


# In[ ]:


get_ipython().system('streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py')


# In[ ]:


get_ipython().system('netstat -tulnp | grep 8501')

