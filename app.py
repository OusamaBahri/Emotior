import streamlit as st
import tensorflow as tf
import numpy as np
img_height = 48
img_width = 48
picture = st.camera_input("Take a picture")
model = tf.keras.models.load_model("EmotionDetecttor2")
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
if picture:
    st.image(picture)

img = tf.keras.utils.load_img(
    picture, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)