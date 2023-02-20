import tensorflow as tf
import streamlit as st

# Load the TensorFlow model
model = tf.keras.models.load_model('keras_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = tf.image.decode_image(image.read(), channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    return image

# Define the cached prediction function
@st.cache_data()
def predict(_image):
    # Make a prediction with the model
    prediction = model.predict(_image)

    # Return the prediction
    return prediction

# Define the Streamlit app
def app():
    # Add a file uploader to the app
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image is uploaded, preprocess it and make a prediction
    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        prediction = predict(image)

        # Display the prediction to the user
        st.write("Prediction:", prediction)

# Run the app
app()
