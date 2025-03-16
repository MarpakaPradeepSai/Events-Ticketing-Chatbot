import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
import subprocess
import os
import sys

# Configuration
MODEL_REPO_URL = "https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot.git"
MODEL_DIR = "./Simple-Events-Ticketing-Customer-Support-Chatbot"
MODEL_PATH = os.path.join(MODEL_DIR, "ALBERT_Model")

# Download model from GitHub if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model files..."):
        try:
            subprocess.run(["git", "clone", "--depth", "1", MODEL_REPO_URL], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Model download failed: {e}")
            st.stop()

# Load SpaCy model with error handling
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    st.error("Downloading SpaCy model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_trf"], check=True)
    nlp = spacy.load("en_core_web_trf")

# Load ALBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cpu")
model.to(device)

# Category mappings (copied from your knowledge base)
category_labels = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 
    3: "check_cancellation_fee", 4: "check_cancellation_policy",
    5: "check_privacy_policy", 6: "check_refund_policy", 7: "customer_service",
    8: "delivery_options", 9: "delivery_period", 10: "event_organizer",
    11: "find_ticket", 12: "find_upcoming_events", 13: "get_refund",
    14: "human_agent", 15: "information_about_tickets", 16: "information_about_type_events",
    17: "pay", 18: "payment_methods", 19: "report_payment_issue",
    20: "sell_ticket", 21: "track_cancellation", 22: "track_refund",
    23: "transfer_ticket", 24: "upgrade_ticket"
}

# Response templates (copied from your knowledge base)
responses = {
    'cancel_ticket': 'To cancel your ticket for the {{EVENT}} in {{CITY}}, please follow these steps:...',
    # ... (include all your response templates here)
}

# Static placeholders (copied from your knowledge base)
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    # ... (include all your static placeholders here)
}

def replace_placeholders(response, dynamic_placeholders):
    """Replace both static and dynamic placeholders in the response"""
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_entities(user_input):
    """Extract EVENT and CITY entities using SpaCy NER"""
    doc = nlp(user_input)
    dynamic_placeholders = {
        '{{EVENT}}': "<b>event</b>",
        '{{CITY}}': "<b>city</b>"
    }
    
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            dynamic_placeholders['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            dynamic_placeholders['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
            
    return dynamic_placeholders

# Streamlit interface
st.set_page_config(page_title="Ticket Support Bot", page_icon="🎫", layout="wide")
st.title("Events Ticketing Customer Support Chatbot")

user_input = st.text_area("Please describe your issue:", height=150)
if st.button("Get Assistance"):
    if user_input.strip():
        # Entity extraction
        dynamic_placeholders = extract_entities(user_input)
        
        # Model inference
        inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        category = category_labels.get(prediction, "Unknown")
        response_template = responses.get(category, "I'm sorry, I couldn't understand your request.")
        
        # Generate final response
        final_response = replace_placeholders(response_template, dynamic_placeholders)
        
        st.markdown("### Assistance Response:")
        st.markdown(final_response, unsafe_allow_html=True)
    else:
        st.warning("Please enter your question first.")
