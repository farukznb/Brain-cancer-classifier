import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime

# Print current working directory for debugging
print("Current working directory:", os.getcwd())

# Custom CNN model definition
class CustomCNN(nn.Module):
    """Optimized Custom CNN model for brain tumor classification."""
    def __init__(self, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Model functions
def load_torch_model(model_path):
    try:
        model = CustomCNN(num_classes=4)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        st.write("PyTorch model loaded successfully.")
        return model
    except Exception as e:
        st.markdown(f'<div class="error-box">Error loading PyTorch model: {str(e)}</div>', unsafe_allow_html=True)
        return None

def load_pb_model(model_path):
    try:
        # Load the model with existing weights, adjust if necessary
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        st.write("TensorFlow model loaded successfully.")
        return model
    except Exception as e:
        st.markdown(f'<div class="error-box">Error loading TensorFlow model: {str(e)}</div>', unsafe_allow_html=True)
        return None

def preprocess_image_torch(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
        ])
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        st.markdown(f'<div class="error-box">Error preprocessing image for PyTorch: {str(e)}</div>', unsafe_allow_html=True)
        return None

def preprocess_image_tf(image):
    try:
        image = image.resize((224, 224))
        image = image.convert("RGB")
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.markdown(f'<div class="error-box">Error preprocessing image for TensorFlow: {str(e)}</div>', unsafe_allow_html=True)
        return None

def predict_torch(model, image):
    if model is None:
        return "Error: PyTorch model not loaded."
    try:
        with torch.no_grad():
            image_tensor = preprocess_image_torch(image)
            if image_tensor is None:
                return "Error: Image preprocessing failed."
            image_tensor = image_tensor.to(torch.device('cpu'))
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            class_names = ["meningioma", "glioma", "pituitary", "notumor"]
            return f"Prediction: {class_names[predicted_class]} (Probability: {probabilities[0][predicted_class]:.4f})"
    except Exception as e:
        st.markdown(f'<div class="error-box">Error during PyTorch prediction: {str(e)}</div>', unsafe_allow_html=True)
        return "Error"

def predict_pb(model, image):
    if model is None:
        return "Error: TensorFlow model not loaded."
    try:
        image_array = preprocess_image_tf(image)
        if image_array is None:
            return "Error: Image preprocessing failed."
        predictions = model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0]
        class_names = ["meningioma", "glioma", "pituitary", "notumor"]
        return f"Prediction: {class_names[predicted_class]} (Probability: {probabilities[predicted_class]:.4f})"
    except Exception as e:
        st.markdown(f'<div class="error-box">Error during TensorFlow prediction: {str(e)}</div>', unsafe_allow_html=True)
        return "Error"

# Function to generate tumor description
def get_tumor_description(predicted_class):
    descriptions = {
        "meningioma": """
            🔅 <b>Meningioma</b>: This tumor originates from the <b>meninges</b>, the protective layers surrounding the brain and spinal cord.<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Usually <b>benign</b> (Grade I), but can be atypical (Grade II) or malignant (Grade III).<br>
                    🔶 Grows slowly, exerting pressure on the brain without invading it.<br>
                    🔶 Symptoms: headaches, vision changes, seizures, or limb weakness.<br>
                    🔶 More common in women; linked to hormones or prior radiation exposure.<br>
            🔅 <b>Treatment</b>: Monitoring, surgery, or radiation therapy if symptomatic.
        """,
        "glioma": """
            🔅 <b>Glioma</b>: This tumor develops from <b>glial cells</b>, which support neurons in the brain.<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Includes subtypes like astrocytoma (Grade I/II or aggressive like glioblastoma Grade IV) and oligodendroglioma.<br>
                    🔶 Infiltrates brain tissue, making complete removal difficult.<br>
                    🔶 Symptoms: headaches, seizures, memory loss, or personality changes.<br>
            🔅 <b>Treatment</b>: Surgery, radiation therapy, chemotherapy, or targeted therapies.
        """,
        "pituitary": """
            🔅 <b>Pituitary Tumor (Pituitary Adenoma)</b>: This tumor forms in the <b>pituitary gland</b>, which controls hormone production at the base of the brain.<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Mostly <b>benign</b> but can disrupt hormone balance.<br>
                    🔶 <b>Functioning tumors</b>: secrete hormones (e.g., prolactinomas cause lactation; ACTH-secreting tumors cause Cushing’s disease).<br>
                    🔶 <b>Non-functioning tumors</b>: cause symptoms by compressing nearby structures (e.g., optic nerves → vision loss).<br>
                    🔶 Symptoms: headaches, fatigue, weight changes, or menstrual irregularities.<br>
            🔅 <b>Treatment</b>: Medications (e.g., dopamine agonists), surgery (via the nose), or radiation therapy.
        """,
        "notumor": """
            🔅 <b>Healthy Brain (No Tumor)</b>: No tumor has been detected in this image.<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Normal structures: <b>neurons</b> (signal transmission) and <b>glial cells</b> (neuron support).<br>
                    🔶 Intact regions: <b>cerebrum</b> (thought, movement), <b>cerebellum</b> (balance), <b>brainstem</b> (breathing, heart rate).<br>
                    🔶 Protective layers: <b>meninges</b> and cerebrospinal fluid (CSF) to protect the brain.<br>
                    🔶 Normal function: cognition, emotions, movement, and hormone regulation intact.<br>
            🔅 <b>Recommendation</b>: Continue regular check-ups to monitor brain health.
        """
    }
    return descriptions.get(predicted_class, "Description not available.")

# Page configuration
st.set_page_config(
    page_title="Brain Cancer Detection",
    page_icon="🧠",
    layout="wide"
)

# CSS styling with transparent buttons and containers
st.markdown("""
    <style>
    body {
        font-family: 'Montserrat', sans-serif;
        background-color: #f5f7fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: 0 auto;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: transparent;
        color: #2c3e50;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: 1px solid transparent;
    }
    .stButton>button:hover {
        background-color: #e8f4f8;
        color: #2980b9;
        border: 1px solid #2980b9;
    }
    .sidebar .stButton>button {
        background-color: transparent;
        color: #2c3e50;
        border: none;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        width: 100%;
        text-align: left;
    }
    .sidebar .stButton>button:hover {
        background-color: #e8f4f8;
        color: #2980b9;
        border: 1px solid #2980b9;
    }
    .logout-button {
        background-color: transparent;
        color: #2c3e50;
        border: none;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        width: 100%;
        text-align: center;
        font-weight: bold;
    }
    .logout-button:hover {
        background-color: #e8f4f8;
        color: #2980b9;
        border: 1px solid #2980b9;
    }
    .stSelectbox {
        background-color: #ecf0f1;
        border-radius: 5px;
        padding: 5px;
    }
    .stFileUploader {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .error-box {
        background-color: #fce4e4;
        color: #c0392b;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .prediction-box {
        background-color: #e8f4f8;
        color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
    }
    .card {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    .transparent-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="background: url('https://example.com/irm-brain-image.jpg') center/cover; opacity: 0.1; position: absolute; width: 100%; height: 100%; z-index: -1;"></div>
    <h1>🧠 Brain Cancer Detection</h1>
""", unsafe_allow_html=True)

# Authentication setup
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def sign_up():
    with st.form("sign_up_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign Up"):
            users = load_users()
            if username in users:
                st.error("Username already exists!")
            else:
                users[username] = {"password": password, "created_at": datetime.now().isoformat()}
                save_users(users)
                st.success("Sign up successful! Please log in.")

def sign_in():
    with st.form("sign_in_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign In"):
            users = load_users()
            if username in users and users[username]["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password!")

# Sidebar with transparent buttons
st.sidebar.title("Menu")
pages = ["What is a Tumor?", "Let's Predict", "Treatment & Advice"]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = pages[0]

for page in pages:
    if st.sidebar.button(page, key=page):
        st.session_state["current_page"] = page

# Authentication logic
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Authentication Required")
    sign_in()
    sign_up()
else:
    # Styled Logout button
    if st.sidebar.button("Logout", key="logout", help="Sign out of your account"):
        del st.session_state["authenticated"]
        del st.session_state["username"]
        st.session_state["current_page"] = pages[0]
        st.experimental_rerun()

    # Page content
    if st.session_state["current_page"] == "What is a Tumor?":
        st.header("What is a Tumor?")
        st.subheader("Detailed Descriptions of Brain Conditions and Healthy Brain")

        # Glioma
        st.markdown("#### 1. Glioma")
        glioma_path = "tumors/Te-gl.jpg"
        if os.path.exists(glioma_path):
            st.image(glioma_path, caption="Glioma MRI", width=300)
        else:
            st.error(f"Image file '{glioma_path}' not found!")
        st.markdown("""
            <div class="transparent-container">
            🔅 <b>Origin</b>: Develops from <b>glial cells</b> (support cells in the brain that help neurons function). Common subtypes include:<br>
                    🔶 <b>Astrocytoma</b> (from astrocytes): Ranges from slow-growing (Grade I/II) to aggressive (Grade III/IV, e.g., glioblastoma).<br>
                    🔶 <b>Oligodendroglioma</b> (from oligodendrocytes): Often affects white matter.<br>
                    🔶 <b>Glioblastoma (GBM)</b>: Highly aggressive (Grade IV).<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Infiltrates brain tissue, making complete removal difficult.<br>
                    🔶 Symptoms depend on location: headaches, seizures, memory loss, or personality changes.<br>
            🔅 <b>Treatment</b>: Surgery, radiation, chemotherapy, or targeted therapies.
            </div>
        """, unsafe_allow_html=True)

        # Meningioma
        st.markdown("#### 2. Meningioma")
        meningioma_path = "tumors/Te-me.jpg"
        if os.path.exists(meningioma_path):
            st.image(meningioma_path, caption="Meningioma MRI", width=300)
        else:
            st.error(f"Image file '{meningioma_path}' not found!")
        st.markdown("""
            <div class="transparent-container">
            🔅 <b>Origin</b>: Arises from the <b>meninges</b> (protective layers around the brain/spinal cord).<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Usually <b>benign</b> (Grade I), but can be atypical (Grade II) or malignant (Grade III).<br>
                    🔶 Grows slowly, pressing on the brain without invading it.<br>
                    🔶 Symptoms: Headaches, vision changes, seizures, or limb weakness.<br>
                    🔶 More common in women; linked to hormones or prior radiation.<br>
            🔅 <b>Treatment</b>: Monitoring, surgery, or radiation if symptomatic.
            </div>
        """, unsafe_allow_html=True)

        # Pituitary Tumor
        st.markdown("#### 3. Pituitary Tumor (Pituitary Adenoma)")
        pituitary_path = "tumors/Te-pi.jpg"
        if os.path.exists(pituitary_path):
            st.image(pituitary_path, caption="Pituitary Tumor MRI", width=300)
        else:
            st.error(f"Image file '{pituitary_path}' not found!")
        st.markdown("""
            <div class="transparent-container">
            🔅 <b>Origin</b>: Forms in the <b>pituitary gland</b> (controls hormone production at the brain’s base).<br>
            🔅 <b>Characteristics</b>:<br>
                    🔶 Mostly <b>benign</b> but can disrupt hormone balance.<br>
                    🔶 <b>Functioning tumors</b> secrete hormones (e.g., prolactinomas cause lactation; ACTH-secreting tumors cause Cushing’s disease).<br>
                    🔶 <b>Non-functioning tumors</b> cause symptoms by compressing nearby structures (e.g., optic nerves → vision loss).<br>
                    🔶 Symptoms: Headaches, fatigue, weight changes, or menstrual irregularities.<br>
            🔅 <b>Treatment</b>: Medication (e.g., dopamine agonists), surgery (via the nose), or radiation.
            </div>
        """, unsafe_allow_html=True)

        # Healthy Brain
        st.markdown("#### 4. Healthy Brain (No Tumor)")
        healthy_path = "tumors/Tr-no.jpg"
        if os.path.exists(healthy_path):
            st.image(healthy_path, caption="Healthy Brain MRI", width=300)
        else:
            st.error(f"Image file '{healthy_path}' not found!")
        st.markdown("""
            <div class="transparent-container">
            🔅 <b>Structure</b>:<br>
                    🔶 <b>Neurons</b>: Nerve cells transmitting signals.<br>
                    🔶 <b>Glial Cells</b>: Support neurons (e.g., astrocytes regulate nutrients; oligodendrocytes insulate nerves).<br>
                     <b>Regions</b>:<br>
                        🔶 <b>Cerebrum</b>: Controls thought, movement, and senses.<br>
                        🔶 <b>Cerebellum</b>: Coordinates balance and movement.<br>
                        🔶 <b>Brainstem</b>: Manages breathing, heart rate, and consciousness.<br>
                     <b>Protective Layers</b>:<br>
                        🔶 <b>Meninges</b> (dura, arachnoid, pia mater) cushion the brain.<br>
                        🔶 <b>Cerebrospinal Fluid (CSF)</b>: Circulates nutrients and removes waste.<br>
                        🔶 <b>Blood-Brain Barrier</b>: Filters harmful substances from the bloodstream.<br>
            🔅 <b>Function</b>:<br>
                    🔶 Normal cognition, emotion, movement, and sensory processing.<br>
                    🔶 No abnormal growths, blockages, or pressure.<br>
                    🔶 Hormones (if pituitary is healthy) are balanced and regulated.
            </div>
        """, unsafe_allow_html=True)

        # Key Differences
        st.markdown("#### Differences")
        st.markdown("""
            <div class="transparent-container">
            🔅 <b>Glioma</b>: Inside brain tissue, often aggressive.<br>
            🔅 <b>Meningioma</b>: On brain’s surface, usually benign.<br>
            🔅 <b>Pituitary Tumor</b>: Affects hormone regulation; symptoms vary by type.<br>
            🔅 <b>Healthy Brain</b>: No tumors, intact structures, and normal function.
            </div>
        """, unsafe_allow_html=True)

    elif st.session_state["current_page"] == "Let's Predict":
        st.header("Let's Predict")
        
        # Add model type selection
        model_type = st.selectbox(
            "Select Model Type",
            ["PyTorch (.pth)", "TensorFlow (.h5)"],
            help="Choose the model to use for prediction."
        )
        st.session_state["model_type"] = model_type

        # Initialize session state for progressive predictions if not exists
        if "predictions" not in st.session_state:
            st.session_state["predictions"] = []

        uploaded_files = st.file_uploader("Upload MRI Images or a Folder", type=["jpg", "png"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)

                if st.button(f"Run Prediction for {uploaded_file.name}", key=uploaded_file.name):
                    if model_type == "PyTorch (.pth)":
                        model_path = "Zainab_best_model.pth"
                        model = load_torch_model(model_path)
                        prediction = predict_torch(model, image)
                        model_used = "PyTorch"
                    elif model_type == "TensorFlow (.h5)":
                        model_path = "Zainab_model.h5"
                        model = load_pb_model(model_path)
                        prediction = predict_pb(model, image)
                        model_used = "TensorFlow"
                    else:
                        st.error("Invalid model type selected.")
                        prediction = "Error"
                        model_used = "N/A"

                    if prediction != "Error":
                        st.markdown(f'<div class="prediction-box">{prediction}</div>', unsafe_allow_html=True)
                        # Extract the predicted class from the prediction
                        predicted_class = prediction.split("Prediction: ")[1].split(" ")[0]
                        # Generate the description
                        description = get_tumor_description(predicted_class)
                        st.markdown(f'<div class="transparent-container">{description}</div>', unsafe_allow_html=True)
                        # Store the full prediction output in session state
                        st.session_state["predictions"].append({
                            "File": uploaded_file.name,
                            "Prediction": prediction,
                            "Description": description,
                            "Model Used": model_used  # Add column for model type
                        })
                    else:
                        st.markdown('<div class="error-box">Error during prediction.</div>', unsafe_allow_html=True)

            if st.session_state["predictions"]:
                st.subheader("Prediction Results")
                df = pd.DataFrame(st.session_state["predictions"])
                st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="brain_cancer_predictions.csv",
                    mime="text/csv"
                )

    elif st.session_state["current_page"] == "Treatment & Advice":
        st.header("Treatment & Advice")
        st.markdown("""
            <div class="transparent-container">
            🔅 <b>General Medical Advice</b>:<br>
                    🔶 Consult a neurologist for personalized treatment plans.<br>
                    🔶 Common treatments include surgery, radiation, and chemotherapy.<br>
                    🔶 Early detection improves outcomes significantly.<br>
            🔅 <b>Sources</b>:<br>
                    🔶 <a href="https://pubmed.ncbi.nlm.nih.gov/">PubMed</a><br>
                    🔶 <a href="https://www.who.int/">World Health Organization (WHO)</a>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 20px;">
        Powered by Farouk Zainab | Brain Cancer Detection © 2025
    </div>
""", unsafe_allow_html=True)