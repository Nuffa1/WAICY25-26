import glob
import streamlit as st
# from sympy import false # Not used, removed for cleanliness
from train_funcs import *
import os
import requests
import torchvision.utils as vutils
from dotenv import load_dotenv
import torch
import cv2
import database as db

# --- CONFIG & SETUP ---
load_dotenv()
st.set_page_config(page_title="Devanagari Scribe", page_icon="🖋️", layout="centered")
db.init_db()

API_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-en-hi"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

# Load Mappings
if os.path.exists("char_to_int.csv"):
    char_to_int = pd.read_csv("char_to_int.csv")
    char_to_int = char_to_int['0'].to_dict()
    char_to_int = {value: key for key, value in char_to_int.items()}
else:
    st.error("char_to_int.csv not found.")
    char_to_int = {}

# --- CSS CODE ---

# --- HELPER FUNCTIONS ---
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        st.error(f"Translation API Error: {e}")
        return []

def crop_white_space(image_tensor: torch.Tensor, padding) -> torch.Tensor:
    if image_tensor.dim() != 4:
        return image_tensor
    is_white = (image_tensor >= 0.97)
    has_white_pixel_in_column = torch.any(is_white, dim=[0, 1, 2]).squeeze()
    first_white_column_indices = torch.nonzero(has_white_pixel_in_column, as_tuple=True)[0]
    if first_white_column_indices.numel() > 0:
        cut_column_index = first_white_column_indices[0].item()
        return image_tensor[:, :, :, padding:cut_column_index - padding]
    else:
        return image_tensor

def in_hindi(input_text, char_list):
    for c in input_text:
        if c not in char_to_int:
            return False
    return True

@st.cache_resource
def load_models():
    """Loads the heavy text encoder and generator models only once."""
    DEVICE = "cpu"
    VOCAB_SIZE = len(char_to_int)
    EMBEDDING_DIM = 256
    CONDITION_DIM = 128
    Z_DIM = 100
    IMG_H = 64
    IMG_W = 256
    IMG_CHANNELS = 1

    text_encoder = TextEncoder(VOCAB_SIZE, EMBEDDING_DIM, CONDITION_DIM, CONDITION_DIM).to(DEVICE)
    gen = Generator(Z_DIM, CONDITION_DIM, IMG_CHANNELS, IMG_H, IMG_W).to(DEVICE)

    try:
        checkpoints = glob.glob("checkpoints/checkpoint_epoch_*.pth")
        if not checkpoints:
            st.error("No checkpoints found in 'checkpoints/' folder.")
            return None, None

        latest_epoch_num = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoints])
        checkpoint_path = f"checkpoints/checkpoint_epoch_{latest_epoch_num}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        text_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        gen.eval()
        text_encoder.eval()
        return text_encoder, gen
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data(show_spinner="Synthesizing Handwriting 🖋️✨")
def generate_handwriting_cached(text_input, char_mapping, max_seq_len=50, noise_seed=42):
    """Generates and returns the concatenated image tensor for a given text."""
    text_encoder, gen = load_models()
    if text_encoder is None:
        return None, text_input

    DEVICE = "cpu"
    Z_DIM = 100
    torch.manual_seed(noise_seed)

    text_list = text_input.split()
    fake_samples = []

    for text in text_list:
        noise = torch.randn(1, Z_DIM, device=DEVICE)
        text_ids = [char_mapping.get(c, char_mapping.get('<UNK>', 0)) for c in text]
        padded_text_ids = torch.zeros(max_seq_len, dtype=torch.long)
        padded_text_ids[:len(text_ids)] = torch.tensor(text_ids)
        padded_text_ids = padded_text_ids.unsqueeze(0).to(DEVICE)

        condition = text_encoder(padded_text_ids)
        fake_sample = gen(noise, condition)
        fake_sample = crop_white_space(fake_sample, 10)
        fake_samples.append(fake_sample)

    if fake_samples:
        return torch.cat(fake_samples, dim=3), text_input
    return None, text_input

# --- SIDEBAR: NAVIGATION ---
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

with st.sidebar:
    if st.session_state['user_id'] is None:
        st.header("Guest Mode")
        st.write("Log in to save your generated sentences.")
        if st.button("Go to Login Page"):
            st.switch_page("pages/Login.py")
    else:
        st.subheader(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout"):
            st.session_state['user_id'] = None
            st.session_state['username'] = None
            st.rerun()

# --- MAIN UI ---
# st.markdown("<h2 style='text-align: center;'>Devanagari Scribe</h2>", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.image("logo.png", use_container_width=True)

st.markdown(
    "<p style='text-align: center;'>Transform English or Hindi text into authentic handwritten calligraphy.</p>",
    unsafe_allow_html=True)
st.divider()

user_input = st.text_input("Enter text to generate:", key="input", placeholder="Type 'Hello' or 'नमस्ते' here...")

if user_input:
    text_to_process = ""

    # 1. Translation Phase
    with st.spinner("Processing text...", show_time=True):
        if not in_hindi(user_input, char_to_int.keys()):
            try:
                api_res = query({"inputs": user_input})
                if isinstance(api_res, list) and len(api_res) > 0:
                    output = api_res[0]['translation_text']
                    st.info(f"**Translated to:** {output}")
                    text_to_process = output
                else:
                    st.error("Translation API returned unexpected format.")
                    text_to_process = user_input
            except Exception:
                text_to_process = user_input
        else:
            text_to_process = user_input

    # 2. Generation Phase
    if text_to_process:
        fake_samples_tensor, final_text = generate_handwriting_cached(text_to_process, char_to_int)

        if fake_samples_tensor is not None:
            with st.spinner("Synthesizing Handwriting...", show_time=True):
                # Save temp file
                vutils.save_image(fake_samples_tensor, "sample.png", normalize=True)

                # Display Image
                st.image("sample.png", )

                # --- ACTION BUTTONS ---
                col1, col2 = st.columns(2)

                with open("sample.png", "rb") as file:
                    col1.download_button(
                        label="Download Image",
                        data=file,
                        file_name=f"devanagari_{text_to_process[:10]}.png",
                        mime="image/png"
                    )

                if st.session_state['user_id']:
                    if col2.button("Save to Profile"):
                        db.save_generated_image(st.session_state['user_id'], text_to_process, "sample.png")
                        st.toast("Image saved to your profile!")
                else:
                    col2.caption("Login to save this image.")