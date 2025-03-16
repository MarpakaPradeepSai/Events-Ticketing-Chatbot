import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
import requests
from io import BytesIO

# Load the ALBERT model and tokenizer from the provided GitHub URL
model_url = "https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model"
tokenizer_url = model_url

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_url)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)

# Load SpaCy NER model for extracting entities
nlp = spacy.load("en_core_web_trf")

# Category labels mapping (same as you provided)
category_labels = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 3: "check_cancellation_fee", 4: "check_cancellation_policy",
    5: "check_privacy_policy", 6: "check_refund_policy", 7: "customer_service", 8: "delivery_options", 9: "delivery_period",
    10: "event_organizer", 11: "find_ticket", 12: "find_upcoming_events", 13: "get_refund", 14: "human_agent", 15: "information_about_tickets",
    16: "information_about_type_events", 17: "pay", 18: "payment_methods", 19: "report_payment_issue", 20: "sell_ticket", 21: "track_cancellation",
    22: "track_refund", 23: "transfer_ticket", 24: "upgrade_ticket"
}

# Responses as you provided
responses = {
    'cancel_ticket': 'To cancel your ticket for the {{EVENT}} in {{CITY}}, please follow these steps:\n\n1. Access {{WEBSITE_URL}}...',
    # Add the rest of your responses here as you already did in the code.
}

# Define static placeholders
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    # Add the rest of your placeholders as you did in the code
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    # Replace both static and dynamic placeholders
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question):
    # Process the user question through SpaCy NER model
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

def get_bot_response(user_question):
    # Extract dynamic placeholders from the user question
    dynamic_placeholders = extract_dynamic_placeholders(user_question)

    # Tokenize the user question
    inputs = tokenizer(user_question, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to("cpu") for key, value in inputs.items()}  # Set to CPU

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    predicted_category_index = prediction.item()
    predicted_category_name = category_labels.get(predicted_category_index, "Unknown Category")

    # Get response from the responses dictionary
    initial_response = responses.get(predicted_category_name, "Sorry, I didn't understand your request. Please try again.")

    # Replace both static and dynamic placeholders in the response
    response = replace_placeholders(initial_response, dynamic_placeholders, static_placeholders)

    return response

# Streamlit UI
st.title("Customer Support Chatbot")
user_question = st.text_input("Ask your question:")

if user_question:
    st.write("**Your Question:**", user_question)
    response = get_bot_response(user_question)
    st.write("**Chatbot Response:**")
    st.write(response)
