import requests
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import os

# Cache the model loading for efficiency
@st.cache_resource
def load_model(model_path="anemia_model.pt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Preprocess image function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Prediction function
def predict_anemia(model, image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
    probability = torch.sigmoid(output).item()
    return ("Anemic", probability) if probability > 0.75 else ("Non-Anemic", probability)

# Load the model
try:
    if "model" not in st.session_state:
        st.session_state.model = load_model("best_model.pt")
    model = st.session_state.model
except Exception as e:
    st.error(f"Error loading model: {str(e)}")


# Custom CSS for improved UI
st.markdown("""
    <style>
     @media screen and (max-width: 768px) {
            .section {
                margin-bottom: 15px;
                padding: 15px;
            }
        }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #F5F7FA 0%, #EDEFF2 100%);
            color: #333333;
        }
        .header {
            text-align: center;
            font-size: 40px;
            font-weight: 700;
            margin: 40px 0 20px;
            color: #D72638;
        }
        .sub-header {
            text-align: center;
            font-size: 18px;
            color: #666666;
            margin-bottom: 40px;
        }
        .section {
            background: white;
            padding: 20px 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 30px;
            max-width: 100%;
        }
        .section h2 {
            font-size: 22px;
            margin-bottom: 15px;
            color: #D72638;
        }
        .upload-preview img {
            border-radius: 10px;
            max-width: 100%;
        }
        .button {
            background: #D72638;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .button:hover {
            background: #A51D2B;
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #ddd;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: #D72638;
            box-shadow: 0 0 6px rgba(215, 38, 56, 0.2);
        }
        .stRadio > label {
            margin-right: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>Iron Deficiency Anemia Test</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-powered anemia detection tool</div>", unsafe_allow_html=True)

# Two-column layout
col_left, col_mid, col_right = st.columns([1.5, 2, 1.2], gap="large")

# Left Column: Photo Guide & Symptoms
with col_left:
    # Photo Guide
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Photo Guide")
    st.markdown("""
        - ✅ Bright, even lighting  
        - ✅ Plain background  
        - ❌ No nail polish  
        - ❌ No obstructions
    """)
    st.image("Non-Anemic.jpeg", caption="Example of a clear image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Symptoms
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Common Symptoms")
    st.markdown("""
        - Pale or sallow skin  
        - Fatigue or low energy  
        - Shortness of breath  
        - Weakness  
        - Rapid heartbeat
    """)
    st.markdown("[Learn more](https://www.hematology.org/education/patients/anemia/iron-deficiency)", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Middle Column: Upload & Analysis (Core Functionality)
with col_mid:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Upload & Analyze")
    
    with st.form(key="analysis_form"):
        name = st.text_input("Your Name", placeholder="John Doe")
        col_age_gender = st.columns(2)
        with col_age_gender[0]:
            age = st.slider("Age", 1, 120, 30)
        with col_age_gender[1]:
            gender = st.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
        medical_history = st.text_area("Medical History (Optional)", placeholder="Previous conditions, medications...")
        
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Clear photo of nails or eyelids")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        submit = st.form_submit_button("Analyze", help="Click to analyze")
    
    if submit:
        if not name.strip() or not uploaded_file:
            st.error("Please enter your name and upload an image.")
        else:
            with st.spinner("Analyzing..."):
                result, confidence = predict_anemia(model, image)
                st.success(f"Result: {result}")
    st.markdown("</div>", unsafe_allow_html=True)

# Right Column: Health Tips
with col_right:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Health Tips")
    st.markdown("""
        - 🥦 Eat iron-rich foods (spinach, lentils)  
        - 🍊 Boost absorption with vitamin C  
        - ☕ Avoid tea/coffee with meals
        - 🥩 Include lean red meat (2-3x/week)
        - 🥣 Cook in cast iron cookware
        - 💊 Consider supplements (consult doctor)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
