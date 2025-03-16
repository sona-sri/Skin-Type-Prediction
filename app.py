import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd

# Define your label mapping
label_index = {0: "dry", 1: "normal", 2: "oily"}
index_label = {0: "dry", 1: "normal", 2: "oily"}

# Define the image transformation
IMG_SIZE = 200
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pretrained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(label_index))
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()  # Set the model to evaluation mode


# Define a prediction function
def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return index_label[predicted.item()]


# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Skin Type Classification</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Layout with two columns for image and classification button
col1, col2 = st.columns(2)

# Only show classify button if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width='never', width=170)
        if st.button("Classify"):
            result = predict(image)
            st.session_state['result'] = result  # Store result in session state

    with col2:
        if 'result' in st.session_state:
            st.markdown("<h5 style='text-align: left;'>Your skin type is predicted successfully!</h5>",
                        unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: left;'>{st.session_state['result'].capitalize()}</h2>",
                        unsafe_allow_html=True)

# Define skincare recommendations based on the predicted skin type
if 'result' in st.session_state:
    recommendations = {
        "dry": {
            "Morning": [
                ("Cleanser", "Gentle Hydrating Cleanser (no acids)"),
                ("Toner", "Hyaluronic Acid (1-2%) for added moisture"),
                ("Serum", "Lactic Acid (5%) for gentle exfoliation"),
                ("Moisturizer", "Rich Emollient Cream (no acids)"),
                ("Sunscreen", "Zinc Oxide or Titanium Dioxide (non-comedogenic)")
            ],
            "Evening": [
                ("Cleanser", "Gentle Hydrating Cleanser (no acids)"),
                ("Toner", "Hyaluronic Acid (1-2%) for added moisture"),
                ("Serum", "Lactic Acid (5%) for gentle exfoliation"),
                ("Moisturizer", "Rich Emollient Cream (no acids)")
            ],
            "Night": [
                ("Cleanser", "Gentle Hydrating Cleanser (no acids)"),
                ("Toner", "Hyaluronic Acid (1-2%) for added moisture"),  # Added toner for night care
                ("Serum", "Lactic Acid (5%) for gentle exfoliation"),
                ("Moisturizer", "Rich Emollient Cream (no acids)")
            ]
        },
        "normal": {
            "Morning": [
                ("Cleanser", "Gentle Cleanser (no acids)"),
                ("Toner", "Rose Water (no acids)"),
                ("Serum", "Niacinamide (5%) for balancing"),
                ("Moisturizer", "Lightweight Hydrating Lotion (no acids)"),
                ("Sunscreen", "Broad Spectrum SPF 30")
            ],
            "Evening": [
                ("Cleanser", "Gentle Cleanser (no acids)"),
                ("Toner", "Rose Water (no acids)"),
                ("Serum", "Niacinamide (5%) for balancing"),
                ("Moisturizer", "Lightweight Hydrating Lotion (no acids)")
            ],
            "Night": [
                ("Cleanser", "Gentle Cleanser (no acids)"),
                ("Toner", "Rose Water (no acids)"),  # Added toner for night care
                ("Serum", "Niacinamide (5%) for balancing"),
                ("Moisturizer", "Lightweight Hydrating Lotion (no acids)")
            ]
        },
        "oily": {
            "Morning": [
                ("Cleanser", "Salicylic Acid Cleanser (2%)"),
                ("Toner", "Glycolic Acid Toner (5%)"),
                ("Serum", "Niacinamide (5%) for oil control"),
                ("Moisturizer", "Oil-Free Moisturizer (no acids)"),
                ("Sunscreen", "Non-comedogenic SPF 30")
            ],
            "Evening": [
                ("Cleanser", "Salicylic Acid Cleanser (2%)"),
                ("Toner", "Glycolic Acid Toner (5%)"),
                ("Serum", "Niacinamide (5%) for oil control"),
                ("Moisturizer", "Oil-Free Moisturizer (no acids)")
            ],
            "Night": [
                ("Cleanser", "Salicylic Acid Cleanser (2%)"),
                ("Toner", "Glycolic Acid Toner (5%)"),  # Added toner for night care
                ("Serum", "Niacinamide (5%) for oil control"),
                ("Moisturizer", "Oil-Free Moisturizer (no acids)")
            ]
        }
    }

    # Get recommendations for the predicted skin type
    skin_type_recommendations = recommendations[st.session_state['result']]

    # Create a structured table for recommendations
    st.markdown(
        f"<h2 style='text-align: center;'>Skincare Recommendations for {st.session_state['result'].capitalize()} Skin</h2>",
        unsafe_allow_html=True)
    table_data = []

    for time_of_day, items in skin_type_recommendations.items():
        table_data.append([time_of_day, "", ""])  # Row for time of day
        for product, description in items:
            table_data.append(["", product, description])  # Product and description

    # Convert to DataFrame for display
    df = pd.DataFrame(table_data, columns=["Time of Day", "Product", "Acid Composition"])

    # Add an index starting from 1
    df.index += 1

    # Display the DataFrame as a table (without scroll view)
    st.table(df)  # Display the DataFrame without index

