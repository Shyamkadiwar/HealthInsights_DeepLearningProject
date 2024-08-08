import streamlit as st
import requests
from PIL import Image
import google.generativeai as genai
import numpy as np
import tensorflow as tf

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Hide default Streamlit elements while keeping the sidebar
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}
    .stDeployButton {display: none;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div:first-child {padding-top: 0rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom button style
custom_button_style = """
<style>
    .stButton > button {
        width: 100%;
        padding: 10px;
        margin: 5px 0;
        color: #EEEEEE;
        background-color: #0D1520;
        border: none;
        border-radius: 5px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #111927;
        transition : width
    }
</style>
"""
st.markdown(custom_button_style, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Home", key="home"):
        st.session_state.page = "Home"
    if st.button("Report Diagnosis", key="ReportDiagnosis"):
        st.session_state.page = "ReportDiagnosis"
    if st.button("Kidney", key="kidney"):
        st.session_state.page = "Kidney"
    if st.button("Pneumonia", key="Pneumonia"):
        st.session_state.page = "Pneumonia"
    if st.button("Cancer", key="cancer"):
        st.session_state.page = "Cancer"

# Main content
page = st.session_state.page

if page == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is the main page of our application.")

elif page == "Cancer":
    AZURE_TRANSLATOR_KEY = "5d807d71150544c79e767709cca00191"
    AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
    AZURE_TRANSLATOR_REGION = "centralindia"

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
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Odia": "or",
        "Assamese": "as",
        "Urdu": "ur",
        "Sanskrit": "sa",
        "Konkani": "kok",
        "Maithili": "mai",
        "Sindhi": "sd",
        "Dogri": "doi",
        "Santali": "sat",
        "Kashmiri": "ks",
        "Nepali": "ne",
        "Manipuri": "mni",
        "Bodo": "brx"
    }

    selected_language = st.selectbox("Select language for response", list(language_codes.keys()))


    API_KEY = "AIzaSyCo4fhOIneKKNTApo7WdkGkAvfSwXQgpf0"
    genai.configure(api_key=API_KEY)

    def get_gemini_response(input, prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input, prompt])
        return response.text

    def load_and_preprocess_image(image_path, target_size):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = np.asarray(img) / 255.0  # Normalize image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    model = tf.keras.models.load_model('C:\shyam\A multiclass disease classifier\saved_model\melanoma_detection_model.h5', compile=False)

    def main():
        st.title('Skin-Cancer Disease Identifier')
        st.write('Upload a skin image for classification')

        uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg"])

        age = st.number_input("Age", min_value=0, max_value=120)
        height = st.number_input("Height (cm)", min_value=50, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=250)

        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            if st.button("Submit"):
                img = load_and_preprocess_image(uploaded_file, target_size=(64, 64))
                prediction = model.predict(img)
                predicted_class = np.argmax(prediction)

                class_labels = {
                    0: 'Actinic keratoses and intraepithelial carcinoma (akiec)', 
                    1: 'Basal cell carcinoma (bcc)', 
                    2: 'Benign keratosis-like lesions (bkl)', 
                    3: 'Dermatofibroma (df)', 
                    4: 'Melanoma (mel)', 
                    5: 'Melanocytic nevi (nv)', 
                    6: 'Vascular lesions (vasc)'
                }
                st.subheader(f"Patient might have: {class_labels[predicted_class]}")

                input = f"user has {class_labels[predicted_class]} disease and age:{age}, height:{height}, and weight:{weight}, so according to this give me a diet plan."
                input_prompts = f"""
                in the first message
                You are an expert dietitian. You will receive input on the user's age, height, weight, and their specific medical condition is {class_labels[predicted_class]}, which in this case is vascular lesions. Based on this information, you will provide general dietary guidance while making sure to include specific considerations for the user's age group. 

                The response should be structured as follows:
                **Dietery plan should include** : //here give the dietery advice according to their height , weight and age give 5-6 points
                **Some exercise** : //suggest some very basic exercise to the patient according to their condition
                **Do's and Dont's** : //suggest few dos and donts pointwise for the patient
                **Some vitamins** : //suggest very basic and general vitamins according to their disease,
                **Disclaimer** : //now at the end you will give the disclaimer that the patient should consult the doctor
                """
                response = get_gemini_response(input, input_prompts)
                if selected_language != "English":
                    translated_response = translate_text(response, language_codes[selected_language])
                    st.subheader(f"Diagnosis Report in {selected_language}:")
                    st.write(translated_response)
                else:
                    st.subheader("Diagnosis Report:")
                    translated_response = translate_text(response, language_codes[selected_language])
                    st.write(translated_response)
                st.subheader(f"The Dietary plan for age {age} is : ")
                st.write(translated_response)
                

    if __name__ == '__main__':
        main()

elif page == "ReportDiagnosis":
    AZURE_TRANSLATOR_KEY = "5d807d71150544c79e767709cca00191"
    AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
    AZURE_TRANSLATOR_REGION = "centralindia"

    API_KEY = "AIzaSyCo4fhOIneKKNTApo7WdkGkAvfSwXQgpf0"
    genai.configure(api_key=API_KEY)

    def get_gemini_response(input, image, prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input, image[0], prompt])

        if response.parts:
            return response.text
        else:
            if response.prompt_feedback.safety_ratings:
                blocked_categories = [
                    rating.category for rating in response.prompt_feedback.safety_ratings
                    if rating.probability != "NEGLIGIBLE"
                ]
                return f"Response was blocked due to safety concerns in the following categories: {', '.join(blocked_categories)}"
            else:
                return "Response was blocked. No specific safety ratings available."

    def input_image_setup(uploaded_file):
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            image_parts = [
                {
                    "mime_type": uploaded_file.type,
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

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

    st.header("Report Diagnostics")

    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Odia": "or",
        "Assamese": "as",
        "Urdu": "ur",
        "Sanskrit": "sa",
        "Konkani": "kok",
        "Maithili": "mai",
        "Sindhi": "sd",
        "Dogri": "doi",
        "Santali": "sat",
        "Kashmiri": "ks",
        "Nepali": "ne",
        "Manipuri": "mni",
        "Bodo": "brx"
    }

    selected_language = st.selectbox("Select language for response", list(language_codes.keys()))

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

    age = st.number_input("Age", min_value=0, max_value=120)
    height = st.number_input("Height (cm)", min_value=50, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=250)

    if st.button("Submit"):
        if uploaded_file is not None:
            image_parts = input_image_setup(uploaded_file)
            input_text = f"User has uploaded the image for diagnosis. User details are age: {age}, height: {height}cm, weight: {weight}kg."
            input_prompts = "You are an expert in image diagnostics. Based on the uploaded image and user details, provide a detailed diagnostic report and relevant dietary advice."
            response = get_gemini_response(input_text, image_parts, input_prompts)

            if selected_language != "English":
                translated_response = translate_text(response, language_codes[selected_language])
                st.subheader(f"Diagnosis Report in {selected_language}:")
                st.write(translated_response)
            else:
                st.subheader("Diagnosis Report:")
                st.write(response)
        else:
            st.warning("Please upload an image for diagnosis.")

elif page == "Kidney":
    import streamlit as st
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import google.generativeai as genai
    import requests

    AZURE_TRANSLATOR_KEY = "5d807d71150544c79e767709cca00191"
    AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
    AZURE_TRANSLATOR_REGION = "centralindia"

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
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Odia": "or",
        "Assamese": "as",
        "Urdu": "ur",
        "Sanskrit": "sa",
        "Konkani": "kok",
        "Maithili": "mai",
        "Sindhi": "sd",
        "Dogri": "doi",
        "Santali": "sat",
        "Kashmiri": "ks",
        "Nepali": "ne",
        "Manipuri": "mni",
        "Bodo": "brx"
    }

    selected_language = st.selectbox("Select language for response", list(language_codes.keys()))



    API_KEY = "AIzaSyCo4fhOIneKKNTApo7WdkGkAvfSwXQgpf0"
    genai.configure(api_key=API_KEY)

## Function to load OpenAI model and get response
    def get_gemini_response(input, prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input, prompt])
        return response.text

# Load model from local file
# @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model('C:\shyam\A multiclass disease classifier\saved_model\kidney_stone_detection_model.h5', compile=False)
        return model

    model = load_model()
# labels = ['Normal', 'Cyst', 'Tumor', 'Stone'] # Update with your actual labels
    labels = ['Cyst', 'Normal', 'Stone', 'Tumor'] # Update with your actual labels

# Streamlit UI
    st.title("Kidney Condition Identifier")
    st.write("Upload a CT scan image of the kidney to predict its condition.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    age = st.number_input("Age", min_value=0, max_value=120)
    height = st.number_input("Height (cm)", min_value=50, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=250)
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(290, 290))
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        # age = st.number_input("Age", min_value=0, max_value=120)
        # height = st.number_input("Height (cm)", min_value=50, max_value=250)
        # weight = st.number_input("Weight (kg)", min_value=10, max_value=250)

        if st.button("Submit"):
            img = image.load_img(uploaded_file, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            st.write("")

            st.subheader(
                "Patient might have {}"
                .format(labels[np.argmax(score)])
            )
            input = f"user has {labels[np.argmax(score)]} disease and age:{age}, height:{height}, and weight:{weight}, so according to this give me a diet plan."
            input_prompts = f"""
            in the first message
            You are an expert dietitian. You will receive input on the user's age, height, weight, and their specific medical condition is {labels[np.argmax(score)]}, which in this case is vascular lesions. Based on this information, you will provide general dietary guidance while making sure to include specific considerations for the user's age group. 
        
            The response should be structured as follows:
            **Dietary plan should include**: //here give the dietary advice according to their height, weight and age give 5-6 points
            **Some exercise**: //suggest some very basic exercise to the patient according to their condition
            **Do's and Dont's**: //suggest few dos and donts pointwise for the patient
            **Some vitamins**: //suggest very basic and general vitamins according to their disease,
            **Disclaimer**: //now at the end you will give the disclaimer that the patient should consult the doctor
            """

            response = get_gemini_response(input, input_prompts)
            if selected_language != "English":
                    translated_response = translate_text(response, language_codes[selected_language])
                    st.subheader(f"Diagnosis Report in {selected_language}:")
                    st.write(translated_response)
            else:
                st.subheader("Diagnosis Report:")
                translated_response = translate_text(response, language_codes[selected_language])
                st.write(translated_response)
            st.subheader(f"The Dietary plan for age {age} is : ")
            st.write(translated_response)
    else:
        st.write("Please upload an image file.")  

elif page == "Pneumonia":
    import streamlit as st
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import torch.nn.functional as F
    from torchvision import models
    import google.generativeai as genai

    AZURE_TRANSLATOR_KEY = "5d807d71150544c79e767709cca00191"
    AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
    AZURE_TRANSLATOR_REGION = "centralindia"

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
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Odia": "or",
        "Assamese": "as",
        "Urdu": "ur",
        "Sanskrit": "sa",
        "Konkani": "kok",
        "Maithili": "mai",
        "Sindhi": "sd",
        "Dogri": "doi",
        "Santali": "sat",
        "Kashmiri": "ks",
        "Nepali": "ne",
        "Manipuri": "mni",
        "Bodo": "brx"
    }

    selected_language = st.selectbox("Select language for response", list(language_codes.keys()))


    API_KEY = "AIzaSyCo4fhOIneKKNTApo7WdkGkAvfSwXQgpf0"
    genai.configure(api_key=API_KEY)

## Function to load OpenAI model and get response
    def get_gemini_response(input, prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input, prompt])
        return response.text

# Define the CustomNeuralNetResNet class
    class CustomNeuralNetResNet(torch.nn.Module):
        def __init__(self, outputs_number=1000):
            super(CustomNeuralNetResNet, self).__init__()
            self.net = models.resnet50(pretrained=False)
            num_ftrs = self.net.fc.in_features
            self.net.fc = torch.nn.Linear(num_ftrs, outputs_number)

        def forward(self, x):
            return self.net(x)

# Load the saved model
    @st.cache_resource
    def load_model():
    # Load the model with 1000 output classes
        model = CustomNeuralNetResNet(1000)
        model.load_state_dict(torch.load('C:\shyam\A multiclass disease classifier\saved_model\pneumonia_classifier_model.h5', map_location=torch.device('cpu')))
    
    # Modify the final layer to have 3 output classes
        num_ftrs = model.net.fc.in_features
        model.net.fc = torch.nn.Linear(num_ftrs, 3)
    
        model.eval()
        return model

# Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Define class names
    class_names = ['NORMAL', 'PNEUMONIA', 'VIRUS']

# Streamlit app
    st.title('Pneumonia Identifier')

# Image upload
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

    age = st.number_input("Age", min_value=0, max_value=120)
    height = st.number_input("Height (cm)", min_value=50, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=250)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded X-ray.', use_column_width=True)

    # Submit button
        if st.button('Predict'):
            model = load_model()

        # Preprocess the image
            input_tensor = transform(image).unsqueeze(0)

        # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()

        # Display results
            st.write(f"Prediction: {class_names[predicted_class]}")
            # st.write("Class probabilities:")
            # for i, prob in enumerate(probabilities):
            #     st.write(f"{class_names[i]}: {prob.item():.2%}")

            input = f"user has {class_names[predicted_class]} disease and age:{age}, height:{height}, and weight:{weight}, so according to this give me a diet plan."
            input_prompts = f"""
            in the first message
            You are an expert dietitian. You will receive input on the user's age, height, weight, and their specific medical condition is {class_names[predicted_class]}, which in this case is vascular lesions. Based on this information, you will provide general dietary guidance while making sure to include specific considerations for the user's age group. 
        
            The response should be structured as follows:
            **Dietary plan should include**: //here give the dietary advice according to their height, weight and age give 5-6 points
            **Some exercise**: //suggest some very basic exercise to the patient according to their condition
            **Do's and Dont's**: //suggest few dos and donts pointwise for the patient
            **Some vitamins**: //suggest very basic and general vitamins according to their disease,
            **Disclaimer**: //now at the end you will give the disclaimer that the patient should consult the doctor
            """

            response = get_gemini_response(input, input_prompts)
            if selected_language != "English":
                    translated_response = translate_text(response, language_codes[selected_language])
                    st.subheader(f"Diagnosis Report in {selected_language}:")
                    st.write(translated_response)
            else:
                st.subheader("Diagnosis Report:")
                translated_response = translate_text(response, language_codes[selected_language])
                st.write(translated_response)
            st.subheader(f"The Dietary plan for age {age} is : ")
            st.write(translated_response)   