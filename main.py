import requests
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import os
# Cache the model loading for efficiency
@st.cache_resource
def load_model(model_path='best_model.pt'):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    print(f"Loading model from {model_path}")  # Debug print
    
    # Initialize model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    
    try:
        # Load in chunks to avoid memory issues
        checkpoint = torch.load(
            model_path,
            map_location='cpu',
            weights_only=False,
            mmap=True  # Memory-map the file
        )
        
        print("Checkpoint loaded, applying to model")  # Debug print
        model.load_state_dict(checkpoint)
        print("State dict applied successfully")  # Debug print
        
    except RuntimeError as e:
        print(f"RuntimeError: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise
        
    model.eval()
    return model


# Preprocess image function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict_anemia(model, image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
    probability = torch.sigmoid(output).item()  # Convert to probability
    return ("Anemic", probability) if probability > 0.5 else ("Non-Anemic", probability)

# Load the model
try:
    if 'model' not in st.session_state:
        print("Loading model into session state")  # Debug print
        st.session_state.model = load_model('best_model.pt')
    model = st.session_state.model
    print("Model loaded successfully")  # Debug print
except Exception as e:
    st.error(f"Error loading model: {str(e)}")


# Streamlit page configuration
# st.set_page_config(
#     page_title="Iron Deficiency Anemia Test System",
#     page_icon="🩸",
#     layout="wide",
# )

# Custom CSS for improved UI
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #FAFAFA;
            color: #333333;
        }
        .header {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 5px;
            color: #D72638;
        }
        .sub-header {
            text-align: center;
            font-size: 20px;
            color: #555555;
            margin-bottom: 20px;
        }
        .section {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .upload-preview img {
            border-radius: 8px;
            max-width: 100%;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #888888;
        }
        .button {
            background-color: #D72638;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
        }
        .button:hover {
            background-color: #A51D2B;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>Iron Deficiency Anemia Test System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-powered tool for quick anemia insights.</div>", unsafe_allow_html=True)

# Three-column layout
col1, col2, col3 = st.columns([1, 2, 1])

# Left Column: Health Tips
with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Health Tips")
    st.markdown("""
        - Include iron-rich foods like spinach and lentils in your diet.
        - Pair iron with vitamin C for better absorption.
        - Drink less tea or coffee with meals to improve iron intake.
    """)
    example_image_path = "Non-Anemic.jpeg"  # Ensure this file is in the working directory
    st.image(example_image_path, caption="Example of a clear image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Center Column: Form and Prediction
with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Input Details and Upload Image")
    
    # User Inputs
    name = st.text_input("Enter your name:", placeholder="Your name here")
    age = st.slider("Age:", min_value=1, max_value=120, value=30)
    gender = st.radio("Gender:", ["Male", "Female", "Other"], horizontal=True)
    medical_history = st.text_area("Medical History:", placeholder="E.g., anemia, recent surgeries")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload an image (nails or eyelids):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze"):
            if not name.strip():
                st.error("Please enter your name.")
            else:
                with st.spinner("Analyzing..."):
                    result, confidence = predict_anemia(model, image)
                    st.success(f"Result: {result}")
                    st.info(f"Confidence Level: {confidence:.4f}")
    else:
        st.warning("Please upload an image to proceed.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Right Column: Guidelines
with col3:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Photo Guidelines")
    st.markdown("""
        - Ensure the photo is clear and well-lit.
        - Remove any obstructions like nail polish.
        - Use a plain background for better results.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2024 Iron Deficiency Anemia Test System | All rights reserved</div>", unsafe_allow_html=True)
