# Text Summarizer Application

This is a text summarization application built with Streamlit and Hugging Face's Transformers library. It uses the T5 model to generate concise summaries of input text.

## Features

- Interactive web interface using Streamlit
- Adjustable summary length
- Real-time statistics about the summarization
- High-quality summaries using the T5 model
- User-friendly design

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your text in the text area and adjust the summary length parameters if desired

4. Click "Generate Summary" to get your summary

## Model Information

This application uses the T5 (Text-to-Text Transfer Transformer) model, specifically the t5-base variant. The model is loaded from Hugging Face's model hub and runs locally on your machine.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Streamlit
- SentencePiece

All dependencies are listed in the `requirements.txt` file. 