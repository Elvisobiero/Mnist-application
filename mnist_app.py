from keras.models import load_model
import streamlit as st
import os
import numpy as np
from PIL import Image
import cv2 as cv

dir = os.getcwd()
model = load_model(os.path.join(dir,'digits.keras'))

def main():
    st.title('MNIST APP')
    st.markdown("Welcome to my MNIST App")

    uploaded_file = st.file_uploader('UPLOAD AN IMAGE')

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='UPLOADED IMAGE')

        if st.button("DETECT"):
            st.write("result...")

            image = np.array(image)
            # image = cv.resize(image, (28, 28))

            # img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            predicted_proba = model.predict(np.expand_dims(image/255, axis=0))
            predicted_class = np.argmax(predicted_proba)

            st.header("Prediction is {}".format(predicted_class))



if __name__ == '__main__':
    main()
