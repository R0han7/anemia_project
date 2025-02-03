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

    print(f"Loading model from {model_path}")  # Debug print

    # Initialize model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    try:
        # Load in chunks to avoid memory issues
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,  # Memory-map the file
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
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
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
    if "model" not in st.session_state:
        print("Loading model into session state")  # Debug print
        st.session_state.model = load_model("best_model.pt")
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
st.markdown(
    """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #FAFAFA 0%, #F0F0F0 100%);
            color: #333333;
            animation: fadeIn 0.8s ease-in;
        }
        .header {
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            margin: 20px 0;
            color: #D72638;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            animation: slideIn 0.8s ease-out;
        }
        .sub-header {
            text-align: center;
            font-size: 20px;
            color: #666666;
            margin-bottom: 30px;
            animation: slideIn 0.8s ease-out 0.2s;
            animation-fill-mode: backwards;
        }
        .section {
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            margin-bottom: 25px;
        }
        .section:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        }
        .upload-preview img {
            border-radius: 12px;
            max-width: 100%;
            transition: transform 0.3s ease;
        }
        .upload-preview:hover img {
            transform: scale(1.02);
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 14px;
            color: #888888;
            animation: fadeIn 1s ease-in 1s backwards;
        }
        .button {
            background: linear-gradient(135deg, #D72638 0%, #A51D2B 100%);
            color: white;
            padding: 12px 28px;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.98); }
            50% { transform: scale(1.02); }
            100% { transform: scale(0.98); }
        }
        .stTextInput>div>div>input {
            border-radius: 8px!important;
            padding: 12px!important;
            border: 2px solid #eee!important;
            transition: all 0.3s ease!important;
        }
        .stTextInput>div>div>input:focus {
            border-color: #D72638!important;
            box-shadow: 0 0 8px rgba(215,38,56,0.2)!important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Header with animated entrance
st.markdown(
    "<div class='header'>Iron Deficiency Anemia Test System</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>AI-powered tool for quick anemia insights</div>",
    unsafe_allow_html=True,
)

# Columns with hover effects
col1, col2, col3 = st.columns([1, 2, 1], gap="large")

# Left Column
with col1:
    st.markdown("<div class='section pulse'>", unsafe_allow_html=True)
    st.header("📌 Health Tips")
    st.markdown(
        """
        <div style="line-height: 1.6;">
            🥦 Include iron-rich foods like spinach and lentils<br>
            🍊 Pair iron with vitamin C for absorption<br>
            ☕ Limit tea/coffee with meals
        </div>
    """,
        unsafe_allow_html=True,
    )
    st.image(
        "Non-Anemic.jpeg", caption="Example of a clear image", use_column_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Center Column
with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("🖼️ Upload & Analyze")

    # Animated form elements
    with st.form(key="analysis_form"):
        name = st.text_input("**Your Name:**", placeholder="John Doe")
        age = st.slider("**Age:**", 1, 120, 30, help="Drag to select your age")
        gender = st.radio(
            "**Gender:**", ["Male", "Female", "Other"], horizontal=True, index=1
        )
        medical_history = st.text_area(
            "**Medical History:**", placeholder="Previous conditions, medications..."
        )

        uploaded_file = st.file_uploader(
            "**Upload Image:**",
            type=["jpg", "jpeg", "png"],
            help="Clear photo of nails or eyelids",
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        submit = st.form_submit_button(
            "Analyze Now", help="Click after filling all fields"
        )

    if submit:
        if not name.strip():
            st.error("Please enter your name.")
        else:
            with st.spinner("Analyzing..."):
                result, confidence = predict_anemia(model, image)
                st.success(f"Result: {result}")
                # st.info(f"Confidence Level: {confidence:.4f}")
    else:
        st.warning("Please upload an image to proceed.")

    st.markdown("</div>", unsafe_allow_html=True)

# Right Column
with col3:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("📸 Photo Guide")
    st.markdown(
        """
        <div style="line-height: 1.6;">
            ✅ Good lighting<br>
            ✅ Plain background<br>
            ❌ No nail polish<br>
            ❌ No obstructions
        </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Animated footer
st.markdown(
    """
    <div class='footer'>
        🩸 Powered by AI Healthcare Technology | © 2025
    </div>
""",
    unsafe_allow_html=True,
)
