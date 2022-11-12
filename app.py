import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

st.title("Bild Klassifikation mit Teachable Machine Modell")
st.info(
    "Teachable Machine Modelle können einfach in eigene Apps wie diese eingebunden werden"
)

img_file_buffer = st.camera_input("Bild aufnehmen")
# img = None
if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)  #
    ##### Teachable Machine Model #####

    # Load the model, prepare data
    # TODO put in the model from teachable machine
    model = load_model("keras_model.h5")
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Run Model
    prediction = model.predict(data)
    # TODO Change these lines to cover the classes used in teachable machine
    prediction_df = pd.DataFrame(
        columns=["Keine Maske", "Maske", "Maske falsch getragen"], data=prediction
    )
    st.write(prediction_df)
    st.warning(prediction_df.idxmax(axis=1)[0])
