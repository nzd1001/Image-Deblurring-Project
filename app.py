import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import random
from models.deblur_unet import DeblurNet
from utils.model import DeblurAdvancedUnet
# --- Model Loading ---
# IMPORTANT: You need to import your model's class here
# from model import YourModelClass  # <--- REPLACE with your model's definition

@st.cache_resource
def load_model(checkpoint_path, device):
    base_model=DeblurNet()
    model = DeblurAdvancedUnet.load_from_checkpoint(checkpoint_path,model=base_model,lr=2e-5)
    model.to(device)
    model.eval()
    return model

def deblur_image(model, image, device):
    """
    Processes a single image with the model.
    """
    # Define the same transformations used during training/inference
    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu()
    
    # Clamp values to the valid [0, 1] range and convert back to a PIL image
    deblurred_img = to_pil(output_tensor.clamp(0, 1))
    return deblurred_img

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Image Deblurring App")

st.title("Image Deblurring with improved Unet model")
st.write("Upload a blurry image or use a random one from the test set ")

# --- Configuration ---
MODEL_PATH = "model_ckp/best.ckpt"  # <--- REPLACE with the path to your .pth file
BLUR_DIR = "data/test/blur"    # <--- REPLACE with your blurry images directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
# This part will run only once and the model will be cached
try:
    model = load_model(MODEL_PATH, DEVICE)
    st.sidebar.success(f"Model loaded successfully on {DEVICE.upper()}!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


if 'blurry_image' not in st.session_state:
    st.session_state.blurry_image = None
if 'deblurred_image' not in st.session_state:
    st.session_state.deblurred_image = None
if 'image_source' not in st.session_state:
    st.session_state.image_source = "None"


# --- Image Selection Sidebar ---
st.sidebar.title("Image Source")
source_option = st.sidebar.radio(
    "Choose an image source:",
    ("Choose a file", "Use a random image"),
    key="source_option_radio" # A unique key is good practice
)

# Option 1: User uploads a file
if source_option == "Choose a file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a blurry image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        # If a new file is uploaded, update the session state
        if st.session_state.image_source != uploaded_file.name:
            st.session_state.blurry_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.deblurred_image = None # Reset deblurred image
            st.session_state.image_source = uploaded_file.name


# Option 2: User wants a random image
else:
    if st.sidebar.button("Load New Random Image"):
        try:
            image_files = [f for f in os.listdir(BLUR_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                random_image_name = random.choice(image_files)
                # Load the new random image into session state
                st.session_state.blurry_image = Image.open(os.path.join(BLUR_DIR, random_image_name)).convert("RGB")
                st.session_state.deblurred_image = None # Reset deblurred image
                st.session_state.image_source = random_image_name
            else:
                st.sidebar.warning(f"No images found in `{BLUR_DIR}`.")
        except Exception as e:
            st.error(f"Could not load random image: {e}")


# --- Main App Logic ---
col1, col2 = st.columns(2)

# Display the original image from session state
if st.session_state.blurry_image is not None:
    with col1:
        st.header("Blur Image")
        st.image(st.session_state.blurry_image, use_container_width=True)

# The Deblur button only performs the action; it doesn't load data
if st.button("✨ Deblur Image", use_container_width=True):
    if st.session_state.blurry_image is not None:
        with st.spinner("Deblurring in progress..."):
            # Perform deblurring and store the result in session state
            st.session_state.deblurred_image = deblur_image(model, st.session_state.blurry_image, DEVICE)
    else:
        st.warning("Please load an image first.")

# Display the deblurred image from session state if it exists
if st.session_state.deblurred_image is not None:
    with col2:
        st.header("Deblurred Image")
        st.image(st.session_state.deblurred_image, use_container_width=True)