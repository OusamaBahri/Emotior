import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
img_height = 48
img_width = 48
img_file_buffer = st.camera_input("Take a picture")
model = tf.keras.models.load_model("EmotionDetecttor_2")
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
if img_file_buffer:
    st.image(img_file_buffer)
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    img_array = tf.keras.utils.img_to_array(img.resize((48,48)))
    img_array = tf.expand_dims(img_array, 0)

#img = tf.keras.utils.load_img( picture, target_size=(img_height, img_width))
# To read image file buffer as a 3D uint8 tensor with TensorFlow:
    
#img_array = tf.keras.utils.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )