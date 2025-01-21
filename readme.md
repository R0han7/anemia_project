# Anemia Detection Project

This project uses deep learning to classify images into two categories: **Anemic** and **Non-Anemic**. The model is trained to automatically detect anemia from facial images by leveraging convolutional neural networks (CNNs) and transfer learning.

## Overview of the Framework

The framework employs **PyTorch**, a popular deep learning library, to train the model. The key steps involved are:

1. **Data Preprocessing**: We process the dataset of images to ensure they are in the right format.
2. **Model Selection**: A pre-trained **ResNet50** model is fine-tuned for binary classification.
3. **Training**: The model is trained on the dataset, learning to distinguish between "Anemic" and "Non-Anemic" images.
4. **Evaluation**: The model is evaluated on a validation dataset to assess its performance.
5. **Optimization**: We use the Adam optimizer and Binary Cross-Entropy Loss to train the model.

## Preprocessing

Before feeding the images into the model, several preprocessing steps are applied:

- **Resize**: All images are resized to **224x224 pixels** to standardize their size.
- **Normalization**: The pixel values are normalized using standard ImageNet values:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`
- **Conversion to Tensor**: The images are converted from a standard image format to a tensor, which the model can process.

## Model

The model used in this project is a pre-trained **ResNet50** model, which is fine-tuned for binary classification. ResNet50 is a deep learning model known for its strong performance in image recognition tasks. We modify the final fully connected layer to output a single value for binary classification ("Anemic" or "Non-Anemic").

### Model Architecture:

- **Pre-trained ResNet50**: Initially trained on the ImageNet dataset, ResNet50 is used as a starting point.
- **Final Layer Modification**: The last fully connected layer is replaced to output a single value (1 output node) for binary classification. The model outputs a probability score, which is passed through a sigmoid function to get a final prediction (0 or 1).

### Loss Function & Optimizer:

- **Loss Function**: **Binary Cross-Entropy with Logits** (`BCEWithLogitsLoss`), which is suitable for binary classification.
- **Optimizer**: **Adam** optimizer is used for training the model with a learning rate of `1e-4`.

## Features Extracted

The ResNet50 model automatically extracts important features from the images, which include:

- **Edges and Boundaries**: Recognizes shapes and contours in the images.
- **Textures and Patterns**: Identifies textures that might indicate skin features or other facial details.
- **Color Information**: Detects subtle changes in skin tone or other visual cues that could be related to anemia.
- **Facial Features**: Detects structures like eyes, nose, and mouth, which may help the model understand overall health conditions.

These learned features are used by the model to distinguish between "Anemic" and "Non-Anemic" images.

## How to Run the Project

1. **Clone the repository** to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


## Requirements

Before running the project, ensure that you have the required dependencies installed. You can install them by using the provided `requirements.txt` file.

### Installation Steps

1. **Clone the repository** to your local machine:

   ```bash
   git clone https://github.com/R0han7/anemia_project.git

python -m venv venv        # Create a virtual environment (optional)
source venv/bin/activate   # Activate the virtual environment (Mac/Linux)
venv/Scripts/activate      # Activate the virtual environment (Windows)
pip install -r requirements.txt

streamlit run main.py


Directory Structure:

│────anemia_project/
    ├── main.py                
├── requirements.txt       
├── readme.md             
