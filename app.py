import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing import image as tf_image
import google.generativeai as genai
import torch.nn.functional as F
from torchvision import models
from streamlit_option_menu import option_menu

# Configure API keys
AZURE_TRANSLATOR_KEY = "5d807d71150544c79e767709cca00191"
AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
AZURE_TRANSLATOR_REGION = "centralindia"

API_KEY = "AIzaSyCo4fhOIneKKNTApo7WdkGkAvfSwXQgpf0"
genai.configure(api_key=API_KEY)

## Function to load OpenAI model and get response
def get_gemini_response(input, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, prompt])
    return response.text

# Function to load and preprocess image for skin cancer model
def load_and_preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.asarray(img) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to translate text using Azure Translator
def translate_text(text, dest_language):
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&to={dest_language}"
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_TRANSLATOR_REGION,
        'Content-Type': 'application/json'
    }

    max_chunk_size = 500
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    translated_chunks = []

    for chunk in chunks:
        body = [{'text': chunk}]
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        translations = response.json()
        translated_chunks.append(translations[0]['translations'][0]['text'])
    
    return ''.join(translated_chunks)

# Load models
@st.cache_resource
def load_skin_cancer_model():
    return tf.keras.models.load_model('saved_model/melanoma_detection_model.h5', compile=False)

@st.cache_resource
def load_kidney_stone_model():
    return tf.keras.models.load_model('saved_model/kidney_stone_detection_model.h5', compile=False)

@st.cache_resource
def load_pneumonia_model():
    # Load the model with 1000 output classes
    model = CustomNeuralNetResNet(1000)
    model.load_state_dict(torch.load('saved_model/pneumonia_classifier_model.h5', map_location=torch.device('cpu')))
    # Modify the final layer to have 3 output classes
    num_ftrs = model.net.fc.in_features
    model.net.fc = torch.nn.Linear(num_ftrs, 3)
    model.eval()
    return model

class CustomNeuralNetResNet(torch.nn.Module):
    def __init__(self, outputs_number=1000):
        super(CustomNeuralNetResNet, self).__init__()
        self.net = models.resnet50(pretrained=False)
        num_ftrs = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(num_ftrs, outputs_number)

    def forward(self, x):
        return self.net(x)

# Define transformations for the pneumonia model
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA', 'VIRUS']
kidney_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
skin_labels = [
    'Actinic keratoses and intraepithelial carcinoma (akiec)', 
    'Basal cell carcinoma (bcc)', 
    'Benign keratosis-like lesions (bkl)', 
    'Dermatofibroma (df)', 
    'Melanoma (mel)', 
    'Melanocytic nevi (nv)', 
    'Vascular lesions (vasc)'
]

# Streamlit UI
st.set_page_config(page_title="Medical Diagnostics and Translation")
st.title('Medical Diagnostics and Translation App')

# Sidebar navigation
selected_app = option_menu(
    "Select the functionality",
    ["Gemini Image Analysis and Translation", "Skin Cancer Classifier", "Kidney Stone Classifier", "Pneumonia Classifier"],
    icons=["images", "person", "heart", "lungs"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Age, Height, and Weight Inputs
age = st.number_input("Age", min_value=0, max_value=120)
height = st.number_input("Height (cm)", min_value=50, max_value=250)
weight = st.number_input("Weight (kg)", min_value=10, max_value=250)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

if selected_app == "Gemini Image Analysis and Translation":
    st.header("Report Diagnostics")

    # Language selection
    language_codes = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Bengali": "bn",
        "Gujarati": "gu", "Marathi": "mr", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa",
        "Odia": "or", "Assamese": "as", "Urdu": "ur", "Sanskrit": "sa", "Konkani": "kok",
        "Maithili": "mai", "Sindhi": "sd", "Dogri": "doi", "Santali": "sat", "Kashmiri": "ks",
        "Nepali": "ne", "Manipuri": "mni", "Bodo": "brx"
    }
    selected_language = st.selectbox("Select language for response", list(language_codes.keys()))

    input_prompt = """
                   You are an expert in understanding invoices.
                   You will receive input images as invoices &
                   you will have to answer questions based on the input image
                   """

    submit = st.button("Analyze and Translate")

    if submit:
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
            response = get_gemini_response(input_prompt, image_parts, """
                    As an expert medical diagnostician:
                    1. Identify the type of medical image/report.
                    2. Describe key observations, focusing on abnormalities.
                    3. List potential conditions suggested by these findings.
                    4. Recommend any necessary follow-up tests.
                    5. Emphasize this is a preliminary analysis requiring professional confirmation.
                    Provide a clear, structured response based solely on visible information.
            """)

            st.subheader("Analysis Result:")
            st.write(response)
            
            if selected_language != "English" and not response.startswith("Response was blocked"):
                translated_response = translate_text(response, language_codes[selected_language])
                st.subheader(f"Translated Response ({selected_language}):")
                st.markdown(translated_response)
        else:
            st.write("Please upload an image first.")

elif selected_app == "Skin Cancer Classifier":
    st.header("Skin-Cancer Disease Identifier")
    st.write('Upload a skin image for classification')

    model = load_skin_cancer_model()

    if uploaded_file is not None and st.button("Submit"):
        img = load_and_preprocess_image(uploaded_file, target_size=(64, 64))
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        st.subheader(f"Patient might have: {skin_labels[predicted_class]}")

        input = f"user has {skin_labels[predicted_class]} disease and age:{age}, height:{height}, and weight:{weight}, so according to this give me a diet plan."
        input_prompts = f"""
        in the first message
        You are an expert dietitian. You will receive input on the user's age, height, weight, and their specific medical condition is {skin_labels[predicted_class]}, which in this case is vascular lesions. Based on this information, you will provide general dietary guidance while making sure to include specific considerations for the user's age group. 

        The response should be structured as follows:
        **Dietary plan should include**: //here give the dietary advice according to their height, weight and age give 5-6 points
        **Some exercise**: //suggest some very basic exercise to the patient according to their condition
        **Do's and Dont's**: //suggest few dos and donts pointwise for the patient
        **Some vitamins**: //suggest very basic and general vitamins according to their disease,
        **Disclaimer**: //now at the end you will give the disclaimer that the patient should consult the doctor
        """

        response = get_gemini_response(input, input_prompts)
        st.write(response)
elif selected_app == "Kidney Stone Classifier":
    st.header("Kidney Disease Identifier")
    st.write('Upload a kidney image for classification')

    model = load_kidney_stone_model()

    if uploaded_file is not None and st.button("Submit"):
        img = load_and_preprocess_image(uploaded_file, target_size=(224, 224))
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        st.subheader(f"Patient might have: {kidney_labels[predicted_class]}")

        input = f"user has {kidney_labels[predicted_class]} disease and age:{age}, height:{height}, and weight:{weight}, so according to this give me a diet plan."
        input_prompts = f"""
        in the first message
        You are an expert dietitian. You will receive input on the user's age, height, weight, and their specific medical condition is {kidney_labels[predicted_class]}, which in this case is vascular lesions. Based on this information, you will provide general dietary guidance while making sure to include specific considerations for the user's age group. 

        The response should be structured as follows:
        **Dietary plan should include**: //here give the dietary advice according to their height, weight and age give 5-6 points
        **Some exercise**: //suggest some very basic exercise to the patient according to their condition
        **Do's and Dont's**: //suggest few dos and donts pointwise for the patient
        **Some vitamins**: //suggest very basic and general vitamins according to their disease,
        **Disclaimer**: //now at the end you will give the disclaimer that the patient should consult the doctor
        """
        response = get_gemini_response(input, input_prompts)
        st.write(response)
elif selected_app == "Pneumonia Classifier":
    st.header("Pneumonia Disease Identifier")
    st.write('Upload a chest x-ray image for classification')

    model = load_pneumonia_model()

    if uploaded_file is not None and st.button("Submit"):
        img = Image.open(uploaded_file).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0]]
        
        st.subheader(f"Patient might have: {predicted_class}")

        input = f"user has {predicted_class} disease and age:{age}, height:{height}, and weight:{weight}, so according to this give me a diet plan."
        input_prompts = f"""
        in the first message
        You are an expert dietitian. You will receive input on the user's age, height, weight, and their specific medical condition is {predicted_class}, which in this case is vascular lesions. Based on this information, you will provide general dietary guidance while making sure to include specific considerations for the user's age group. 

        The response should be structured as follows:
        **Dietary plan should include**: //here give the dietary advice according to their height, weight and age give 5-6 points
        **Some exercise**: //suggest some very basic exercise to the patient according to their condition
        **Do's and Dont's**: //suggest few dos and donts pointwise for the patient
        **Some vitamins**: //suggest very basic and general vitamins according to their disease,
        **Disclaimer**: //now at the end you will give the disclaimer that the patient should consult the doctor
        """

        response = get_gemini_response(input, input_prompts)
        st.write(response)
