import numpy as np
from tensorflow import keras
from PIL import Image
import streamlit as st
import keras.utils as image
import warnings
import base64
import PIL
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
import io


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

encoded_image = get_img_as_base64(r"C:\Users\Hp\Desktop\tomatoyolov8\tomato2.jpg")



page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpg;base64,{encoded_image}");
background-size: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{

background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
# loading in the model to predict on the data
Image_Width = 256
Image_Height = 256
Image_Size = (Image_Width, Image_Height)

classifier = keras.models.load_model('new_tomato.h5', compile=False)
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #000000;
    }
    .big-font {
        font-size:40px !important;
        text-align: center;
        color: black;
        font-style:Helvetica;
        font-weight: bold;
        color:#ffffff;


    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Tomato Disease Prediction</p>', unsafe_allow_html=True)
# here we define some of the front end elements of the web page like
# the font and background color, the padding and the text to be displayed
with st.sidebar:
    st.image("https://www.linkpicture.com/q/garden-plant-in-hand-cartoon-vector-24382370_1_-removebg-preview.png")


    st.markdown("""
    <style>
   
    .big-font {
        font-size:40px !important;
        text-align: center;
        color: black;
        font-style:Helvetica;
        font-weight: bold;
        color:#ffffff;


    }
    
    .small-font {
        font-size:15px !important;
        text-align: center;
        color: purple;
        font-style:Helvetica;
        font-weight: bold;
        color:#ffffff;


    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Podha</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Developed by:<br>Harshit Pokhriyal</br></p>', unsafe_allow_html=True)

# }
results = {
    0: 'Tomato Early Blight',
    1: 'Tomato Late Blight',
    2: 'Tomato Leaf Mold',
    3: 'Tomato Healthy',

}


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('tomato.jpg');
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def preprocess_image(image):
    # Resize the image to the input size required by the model
    resized_image = image.resize((256, 256))
    # Convert the image to a NumPy array
    image_array = np.array(resized_image)
    # Normalize the image array
    normalized_image = image_array / 255.0
    # Expand the dimensions of the image to match the model's input shape
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

def predict(image):
    # Preprocess the image
    input_image = preprocess_image(image)
    # Make the prediction using the loaded model
    predictions = classifier.predict(input_image)
    # Get the predicted class label
    predicted_label = results[np.argmax(predictions)]
    # Get the confidence score for the predicted class
    confidence = np.max(predictions) * 100
    return predicted_label, confidence

# this is the main function in which we define our webpage
def main():

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open and display the uploaded image
        image = PIL.Image.open(uploaded_image)
        # Reduce the image size for preview
        image = image.resize((150, 150))

        # Display the image and prediction side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image')
        with col2:
            # Check if 'Predict' button is clicked
            if st.button('Predict'):
                # Make predictions
                predicted_label, confidence = predict(image)

                # Display the predicted class label and confidence score
                st.write("Prediction:", predicted_label)
                st.write("Confidence:", confidence, "%")





if __name__ == '__main__':
    add_logo()
    main()