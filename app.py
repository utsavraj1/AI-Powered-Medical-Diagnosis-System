import streamlit as st
import pickle
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# 🔹 Configure Dataset Path (Change this as per your system)
DATASET_PATH = DATASET_PATH = r"C:\Users\utsha\Desktop\Medical Diagosis\Medical diagnosis using AI\Datasets\Blood_Cell_Cancer_image_data"
TRAIN_PATH = r"C:\Users\utsha\Desktop\Medical Diagosis\Medical diagnosis using AI\Datasets\Blood_Cell_Cancer_image_data\train"
TEST_PATH = r"C:\Users\utsha\Desktop\Medical Diagosis\Medical diagnosis using AI\Datasets\Blood_Cell_Cancer_image_data\test"

# ✅ Load the Pre-Trained Model
blood_cancer_model = tf.keras.models.load_model("Models/blood_cancer_model.h5")

print("✅ Blood Cancer Model Loaded Successfully!")

# 🔹 Configure Streamlit Page
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# 🔹 Hide Streamlit UI extras
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# 🔹 Set Background Image
BACKGROUND_IMAGE_URL = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url({BACKGROUND_IMAGE_URL});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 🔹 Load Trained Models
models = {
    "blood_cancer": tf.keras.models.load_model("Models/blood_cancer_model.h5"),
    "diabetes": pickle.load(open("Models/diabetes_model.sav", "rb")),
    "heart_disease": pickle.load(open("Models/heart_disease_model.sav", "rb")),
    "parkinsons": pickle.load(open("Models/parkinsons_model.sav", "rb")),
    "lung_cancer": pickle.load(open("Models/lungs_disease_model.sav", "rb")),
    "thyroid": pickle.load(open("Models/Thyroid_model.sav", "rb")),
}

# 🔹 Create Disease Prediction Dropdown
selected = st.selectbox(
    "Select a Disease to Predict",
    [
        "Blood Cancer Prediction",
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Parkinsons Prediction",
        "Lung Cancer Prediction",
        "Hypo-Thyroid Prediction",
    ],
)

# 🔹 Helper Function for User Input
def display_input(label, tooltip, key, type="text"):
    if type == "text":
        return st.text_input(label, key=key, help=tooltip)
    elif type == "number":
        return st.number_input(label, key=key, help=tooltip, step=1)


# 🔹 Blood Cancer Prediction Section
if selected == "Blood Cancer Prediction":
    st.title("Blood Cell Cancer Prediction")
    st.write("Upload a blood cell image to check for cancer.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess Image
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # Make Prediction
            prediction = models["blood_cancer"].predict(image)
            result = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected"

            st.success(result)

        except Exception as e:
            st.error(f"Error processing image: {e}")


# 🔹 Diabetes Prediction Section
if selected == "Diabetes Prediction":
    st.title("Diabetes")
    st.write("Enter the following details to predict diabetes:")

    Pregnancies = display_input("Number of Pregnancies", "Enter number of times pregnant", "Pregnancies", "number")
    Glucose = display_input("Glucose Level", "Enter glucose level", "Glucose", "number")
    BloodPressure = display_input("Blood Pressure value", "Enter blood pressure value", "BloodPressure", "number")
    SkinThickness = display_input("Skin Thickness value", "Enter skin thickness value", "SkinThickness", "number")
    Insulin = display_input("Insulin Level", "Enter insulin level", "Insulin", "number")
    BMI = display_input("BMI value", "Enter Body Mass Index value", "BMI", "number")
    DiabetesPedigreeFunction = display_input("Diabetes Pedigree Function value", "Enter diabetes pedigree function value", "DiabetesPedigreeFunction", "number")
    Age = display_input("Age of the Person", "Enter age of the person", "Age", "number")

    if st.button("Diabetes Test Result"):
        diab_prediction = models["diabetes"].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
        st.success(diab_diagnosis)
