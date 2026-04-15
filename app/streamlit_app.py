import streamlit as st
import torch
import yaml
import os
import sys

# This ensures the app can find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import NextWordLSTM
from src.utils import load_vocab, generate_text

st.set_page_config(page_title="Next Word Predictor", page_icon="🔮", layout="wide")

st.title("🔮 Advanced Next Word Predictor (LSTM)")
st.caption("9+ Rated Project | WandB Tracked | Top-k/Top-p Sampling")

@st.cache_resource
def load_trained_model():
    # Load config to get model dimensions
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    word_to_idx, idx_to_word, vocab_size = load_vocab()
    
    # Initialize model with the SAME settings used in training
    model = NextWordLSTM(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"]
    )
    
    # Load the "brain" we are currently training
    if os.path.exists("models/best_model.pt"):
        model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
    model.eval()
    return model, word_to_idx, idx_to_word

# Load everything
model, word_to_idx, idx_to_word = load_trained_model()

# UI Layout
col1, col2 = st.columns(2)

with col1:
    prompt = st.text_input("Enter a Shakespearean start:", "To be or not")
    max_tokens = st.slider("Words to generate", 5, 100, 30)

with col2:
    temp = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.8)
    top_k = st.number_input("Top-K Sampling", value=10)

if st.button("Generate Next Words"):
    with st.spinner("Thinking like Shakespeare..."):
        # We use the generate_text function from your utils.py
        result = generate_text(
            model, word_to_idx, idx_to_word, 
            prompt=prompt, 
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=top_k
        )
        st.markdown(f"### Result:\n**{result}**")