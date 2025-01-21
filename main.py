# import requests  # Add this to handle API calls
# import streamlit as st
# from PIL import Image
# import torch
# from torchvision import transforms, models
# from PIL import Image
# import streamlit as st

# # Function to load the trained model
# def load_model(model_path='best_model.pth'):
#     model = models.resnet50(pretrained=True)
#     model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Output size = 1 for binary classification
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Adjusted for CPU usage
#     model.eval()  # Set the model to evaluation mode
#     return model

# # Function to preprocess the image
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = transform(image)
#     image = image.unsqueeze(0)  # Add a batch dimension
#     return image

# # Function to make predictions
# def predict_anemia(model, image):
#     image = preprocess_image(image)
#     with torch.no_grad():
#         output = model(image)  # Get the model's output
#     probability = torch.sigmoid(output).item()  # Convert to probability
#     if probability > 0.5:
#         return "Anemic", probability
#     else:
#         return "Non-Anemic", probability

# # Load the model once when the app starts
# model_path = "best_model.pth"  # Replace with your model path
# model = load_model(model_path)

# # Set up page configuration
# st.set_page_config(
#     page_title="Iron Deficiency Anemia Test System",
#     page_icon="🩸",
#     layout="wide",
# )

# # Custom CSS for styling with light and dark mode
# st.markdown(
#     """
#     <style>
#         :root {
#             --main-bg-color: #FAFAFA;
#             --section-bg-color: #FFFFFF;
#             --text-color: #000000;
#             --accent-color: #D72638;
#             --subtle-text-color: #7D7D7D;
#             --guidelines-bg-color: #FFF5F5;
#             --guidelines-border-color: #D72638;
#             --button-bg-color: #D72638;
#             --button-hover-bg-color: #A51D2B;
#         }

#         @media (prefers-color-scheme: dark) {
#             :root {
#                 --main-bg-color: #1E1E1E;
#                 --section-bg-color: #2C2C2C;
#                 --text-color: #FFFFFF;
#                 --accent-color: #FF6B6B;
#                 --subtle-text-color: #B0B0B0;
#                 --guidelines-bg-color: #332222;
#                 --guidelines-border-color: #FF6B6B;
#                 --button-bg-color: #FF6B6B;
#                 --button-hover-bg-color: #D72638;
#             }
#         }

#         body {
#             font-family: 'Arial', sans-serif;
#             background-color: var(--main-bg-color);
#             color: var(--text-color);
#         }
#         .main-header {
#             text-align: center;
#             font-size: 50px;
#             font-weight: bold;
#             color: var(--accent-color);
#             margin-bottom: 10px;
#             background: linear-gradient(to right, var(--accent-color), var(--guidelines-border-color));
#             -webkit-background-clip: text;
#             -webkit-text-fill-color: transparent;
#         }
#         .sub-header {
#             text-align: center;
#             font-size: 22px;
#             color: var(--subtle-text-color);
#             margin-bottom: 30px;
#         }
#         .section {
#             background-color: var(--section-bg-color);
#             padding: 25px;
#             border-radius: 12px;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
#         }
#         .guidelines {
#             background-color: var(--guidelines-bg-color);
#             padding: 20px;
#             border-radius: 10px;
#             border-left: 5px solid var(--guidelines-border-color);
#         }
#         .guidelines h4 {
#             margin-top: 0;
#         }
#         .upload-preview img {
#             border-radius: 8px;
#             max-width: 100%;
#         }
#         .submit-btn {
#             background-color: var(--button-bg-color);
#             color: white;
#             padding: 10px 20px;
#             border-radius: 8px;
#             font-size: 18px;
#             font-weight: bold;
#             transition: background-color 0.3s;
#         }
#         .submit-btn:hover {
#             background-color: var(--button-hover-bg-color);
#         }
#         .footer {
#             text-align: center;
#             margin-top: 30px;
#             font-size: 14px;
#             color: var(--subtle-text-color);
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Main Header
# st.markdown("<div class='main-header'>Iron Deficiency Anemia Test System</div>", unsafe_allow_html=True)
# st.markdown(
#     "<div class='sub-header'>Get quick insights into your iron levels with our AI-powered testing tool.</div>",
#     unsafe_allow_html=True,
# )

# # API Endpoint URL
# API_URL = "http://127.0.0.1:8000/predict"  # Replace with your API endpoint if different

# # Create a three-column layout
# col1, col2, col3 = st.columns([1, 2, 1])

# # Left Column: Additional Tips or Branding
# with col1:
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#     st.header("Health Tips")
#     st.markdown(
#         """
#         - Eat iron-rich foods like spinach, lentils, and red meat.
#         - Pair iron intake with vitamin C for better absorption.
#         - Consult your doctor if symptoms persist.
#         """
#     )
    
#     # Display the example photo
#     photo_path = "Non-Anemic.jpeg"  # Ensure this file is in the working directory or provide the full path
#     st.image(photo_path, caption="Upload a photo like this", use_column_width=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Center Column: Input Form
# with col2:
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#     st.header("Step 1: Enter Your Details")
    
#     # User Details Form
#     age = st.slider("Age:", min_value=1, max_value=120, value=30, step=1)
#     gender = st.radio("Gender:", ["Male", "Female", "Other"], horizontal=True)
#     medical_history = st.text_area("Do you have any medical history?", placeholder="E.g., anemia, recent surgeries")

#     st.markdown("<hr>", unsafe_allow_html=True)
    
#     # Image Upload with Preview
#     st.header("Step 2: Upload Image for Analysis")
#     name = st.text_input("Enter your name:", placeholder="Your name here")

#     uploaded_file = st.file_uploader(
#         "Upload a clear image of your nails or eyelids for assessment:", type=["jpg", "jpeg", "png"]
#     )
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB if needed
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         if st.button("Analyze"):
#             if not name.strip():
#                 st.error("Please enter your name before proceeding.")

#             with st.spinner("Analyzing..."):
#                 label, probability = predict_anemia(model, image)
#                 st.success(f"Prediction: {label}")
#                 st.info(f"Confidence: {probability:.4f}")
#     else:
#         st.warning("Please upload an image to proceed.")
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Right Column: Guidelines
# with col3:
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#     st.header("Photo Guidelines")
#     st.markdown(
#         """
#         <div class='guidelines'>
#             <h4>Ensure accurate results by following these steps:</h4>
#             <ul>
#                 <li>Use natural or bright indoor lighting.</li>
#                 <li>Remove nail polish, makeup, or obstructions.</li>
#                 <li>Ensure the image is focused and steady.</li>
#                 <li>Use a plain background for clarity.</li>
#             </ul>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

# # Footer
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown("<div class='footer'>© 2024 Iron Deficiency Anemia Test System | All rights reserved</div>", unsafe_allow_html=True)
import requests
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models

# Cache the model loading for efficiency
@st.cache_resource
def load_model(model_path='best_model.pth'):
    
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Output size = 1 for binary classification
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=False,encoding="ascii"))  # Adjusted for CPU usage
    model.eval()  # Set the model to evaluation mode
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
model_path = "best_model.pth"
model = load_model(model_path)

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
