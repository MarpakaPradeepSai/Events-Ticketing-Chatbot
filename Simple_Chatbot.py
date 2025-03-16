import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
from spacy.cli import download

# Load SpaCy model for NER with fallback
def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        download(model_name)
        return spacy.load(model_name)

nlp = load_spacy_model("en_core_web_trf")

# Load model and tokenizer from GitHub repo
model_path = "https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Device configuration
device = torch.device("cpu")
model = model.to(device)
model.eval()

# Category labels and responses (same as your original code)
category_labels = {0: "buy_ticket", 1: "cancel_ticket", ...}  # Copy your full category_labels dict
responses = {'cancel_ticket': '...', ...}  # Copy your full responses dict
static_placeholders = {'{{WEBSITE_URL}}': '...', ...}  # Copy your full static_placeholders

# Helper functions (same as original)
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    # Your existing implementation
    return response

def extract_dynamic_placeholders(user_question):
    # Your existing implementation
    return dynamic_placeholders

# Streamlit UI
st.title("🎫 Event Ticketing Customer Support")
st.write("Ask me anything about tickets, events, payments, or cancellations!")

user_input = st.text_input("Type your question here:")

if user_input:
    with st.spinner("Analyzing your question..."):
        # Extract entities
        dynamic_placeholders = extract_dynamic_placeholders(user_input)
        
        # Model prediction
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
        category = category_labels.get(prediction, "general_inquiry")

        # Generate response
        response_template = responses.get(category, responses['customer_service'])
        formatted_response = replace_placeholders(response_template, dynamic_placeholders, static_placeholders)

    st.subheader("🤖 Bot Response")
    st.markdown(formatted_response)
    
    st.subheader("🔍 Analysis Details")
    st.write(f"**Identified Category:** {category.replace('_', ' ').title()}")
    if dynamic_placeholders:
        st.write("**Detected Details:**")
        for key, value in dynamic_placeholders.items():
            st.write(f"- {key.strip('{}')}: {value.strip('<>')}")
