import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os

# Set page config
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Initialize the model and tokenizer
@st.cache_resource
def load_model():
    try:
        model_name = "t5-base"
        # Show a message while downloading
        with st.spinner("Downloading model (this may take a few minutes the first time)..."):
            tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="./model_cache")
            model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="./model_cache")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check your internet connection and try again.")
        return None, None

def summarize_text(text, max_length=150, min_length=40):
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Model failed to load. Please refresh the page and try again.")
        return None
    
    try:
        # Prepare the text for summarization
        input_text = f"summarize: {text}"
        
        # Tokenize the input text
        inputs = tokenizer.encode(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Streamlit UI
st.title("📝 Text Summarizer")
st.write("Enter your text below and get an AI-generated summary!")

# Text input
text_input = st.text_area(
    "Enter your text here:",
    height=300,
    placeholder="Paste your text here..."
)

# Summary length controls
col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("Maximum summary length", 50, 200, 150)
with col2:
    min_length = st.slider("Minimum summary length", 10, 100, 40)

# Generate summary button
if st.button("Generate Summary"):
    if text_input:
        with st.spinner("Generating summary..."):
            summary = summarize_text(text_input, max_length, min_length)
            
            if summary:
                # Display the summary
                st.subheader("Generated Summary")
                st.write(summary)
                
                # Display statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Length", f"{len(text_input.split())} words")
                with col2:
                    st.metric("Summary Length", f"{len(summary.split())} words")
                
                # Compression ratio
                compression_ratio = len(summary.split()) / len(text_input.split()) * 100
                st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
    else:
        st.warning("Please enter some text to summarize.")

# Add some information about the model
with st.expander("About this Summarizer"):
    st.write("""
    This text summarizer uses the T5 (Text-to-Text Transfer Transformer) model, 
    specifically the t5-base variant. T5 is a powerful language model that can 
    perform various text-to-text tasks, including summarization.
    
    Features:
    - Adjustable summary length
    - Maintains key information from the original text
    - Fast and efficient processing
    - High-quality summaries
    
    The model is loaded from Hugging Face's model hub and runs locally on your machine.
    """)
