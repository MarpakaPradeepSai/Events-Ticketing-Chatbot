import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")  # Using smaller model for faster loading

# Load model and tokenizer from GitHub
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model"  # Local folder name in your GitHub repo
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()
device = torch.device("cpu")
model.to(device)
model.eval()

# Category labels mapping (copied from your original code)
category_labels = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 
    # ... (keep all your original mappings)
    24: "upgrade_ticket"
}

# Responses and static placeholders (copied from your original code)
responses = { ... }  # Keep your original responses dictionary
static_placeholders = { ... }  # Keep your original static placeholders

def replace_placeholders(response, dynamic_placeholders):
    # First replace static placeholders
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    # Then replace dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            dynamic_placeholders['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            dynamic_placeholders['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
    # Add defaults if not found
    dynamic_placeholders.setdefault('{{EVENT}}', 'event')
    dynamic_placeholders.setdefault('{{CITY}}', 'city')
    return dynamic_placeholders

# Streamlit interface
st.title("Events Ticketing Support Chatbot")
st.write("Ask any questions about ticket purchases, cancellations, or event information:")

user_question = st.text_input("Your question:")
if st.button("Get Response") and user_question:
    with st.spinner("Processing..."):
        # Extract dynamic placeholders
        dynamic_placeholders = extract_dynamic_placeholders(user_question)
        
        # Tokenize input
        inputs = tokenizer(user_question, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        predicted_category = category_labels.get(prediction, "Unknown")
        initial_response = responses.get(predicted_category, "Sorry, I couldn't understand your request.")
        
        # Replace placeholders
        final_response = replace_placeholders(initial_response, dynamic_placeholders)
        
        # Display response
        st.markdown("### Response:")
        st.markdown(final_response, unsafe_allow_html=True)
