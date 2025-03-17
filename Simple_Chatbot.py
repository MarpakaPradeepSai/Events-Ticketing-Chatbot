import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy

# Load the en_core_web_trf model
@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Initialize the model
nlp = load_model()

# Path to the fine-tuned model (relative to the root of the GitHub repo)
model_path = "ALBERT_Model"  # Matches your repo structure: /ALBERT_Model/

# Load the fine-tuned model and tokenizer
@st.cache_resource  # Cache to avoid reloading on every interaction
def load_model_and_tokenizer():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

# Check if model loaded successfully
if model is None or tokenizer is None:
    st.stop()  # Halt execution if model loading fails

# Set device to CPU (Streamlit Cloud typically doesn't provide GPUs)
device = torch.device("cpu")
model.to(device)

# Category labels mapping (from your original code)
category_labels = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 3: "check_cancellation_fee", 4: "check_cancellation_policy",
    5: "check_privacy_policy", 6: "check_refund_policy", 7: "customer_service", 8: "delivery_options", 9: "delivery_period",
    10: "event_organizer", 11: "find_ticket", 12: "find_upcoming_events", 13: "get_refund", 14: "human_agent", 15: "information_about_tickets",
    16: "information_about_type_events", 17: "pay", 18: "payment_methods", 19: "report_payment_issue", 20: "sell_ticket", 21: "track_cancellation",
    22: "track_refund", 23: "transfer_ticket", 24: "upgrade_ticket"
}

# Response templates (from your original code)
responses = {
    'cancel_ticket': 'To cancel your ticket for the {{EVENT}} in {{CITY}}, please follow these steps:\n\n1. Access {{WEBSITE_URL}} and sign in to your account.\n2. Go to the {{CANCEL_TICKET_SECTION}} section.\n3. Locate your upcoming events and click on the {{EVENT}} in {{CITY}}.\n4. Select the {{CANCEL_TICKET_OPTION}} option.\n5. Complete the prompts to finalize your cancellation.\n\nIf any issues arise, do not hesitate to reach out to our customer support for further help.',
    'buy_ticket': "To acquire a ticket for the {{EVENT}} in {{CITY}}, please undertake the following steps:\n\n1. Access {{WEBSITE_URL}} or launch the {{APP}}.\n2. Proceed to the {{TICKET_SECTION}} segment.\n3. Input the specifics of the desired event or performance.\n4. Identify and select the event from the listed search results.\n5. Specify the quantity of tickets and choose preferred seating arrangements (if applicable).\n6. Move to the checkout phase and provide the required payment details.\n\nUpon completion of your purchase, you will receive an email confirmation containing your ticket information.",
    'change_personal_details_on_ticket': 'To update your personal details on your ticket, please adhere to the following steps:\n\n1. Go to {{WEBSITE_URL}} and sign in to your account.\n2. Proceed to the {{TICKET_SECTION}} section.\n3. Choose the specific ticket for the event in {{CITY}} that you want to amend.\n4. Click on the {{EDIT_BUTTON}} icon next to your personal details.\n5. Make the necessary amendments to your personal information.\n6. Confirm the changes by clicking {{SAVE_BUTTON}}.\n\nIf you face any challenges, please contact our customer support using the contact form available on the website.',
    'check_cancellation_fee': 'To verify the cancellation fee, kindly adhere to these steps:\n\n1. Access the {{WEBSITE_URL}}.\n2. Proceed to the {{CANCELLATION_FEE_SECTION}} segment.\n3. Identify the cancellation fee information in the {{CHECK_CANCELLATION_FEE_INFORMATION}} area.\n\nIf you require additional help, do not hesitate to reach out.',
    'check_cancellation_policy': "For comprehensive details regarding our cancellation policy, please follow these instructions: \n\n1. Access our official website via the following link: {{WEBSITE_URL}}.\n2. Locate the {{CANCELLATION_POLICY_SECTION}} section within the main navigation menu.\n3. Select the {{CHECK_CANCELLATION_POLICY_OPTION}} option from the subsequent dropdown menu.\n4. Carefully read through the cancellation policy presented on the specific page.\n\nIf you require any additional information or have more inquiries, please do not hesitate to reach out to our customer support team for further assistance."
}

# Static placeholders (from your original code, expanded for completeness)
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}": "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{APP}}": "Events App",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fees</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Details</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>View Cancellation Policy</b>"
}

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question):
    # Process the user question through SpaCy NER model
    doc = nlp(user_question)

    # Initialize dictionary to store dynamic placeholders
    dynamic_placeholders = {}

    # Extract entities and map them to placeholders
    for ent in doc.ents:
        if ent.label_ == "EVENT":  # Assuming 'EVENT' is the label for event names (customize based on your model)
            event_text = ent.text.title()  # Capitalize the first letter of each word in the event name
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"  # Bold the entity
        elif ent.label_ == "GPE":  # GPE is the label for cities in SpaCy
            city_text = ent.text.title()  # Capitalize the first letter of each word in the city
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"  # Bold the entity

    # If no event or city was found, add default values
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"

    return dynamic_placeholders

# Streamlit UI
st.title("Simple Events Ticketing Chatbot")
st.write("Ask me anything about ticketing for your events!")

# User input
user_question = st.text_input("Enter your question:", key="user_input")

if user_question:
    # Extract dynamic placeholders
    dynamic_placeholders = extract_dynamic_placeholders(user_question)

    # Tokenize input
    inputs = tokenizer(user_question, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    predicted_category_index = prediction.item()
    predicted_category_name = category_labels.get(predicted_category_index, "Unknown Category")

    # Get and format response
    initial_response = responses.get(predicted_category_name, "Sorry, I didn't understand your request. Please try again.")
    response = replace_placeholders(initial_response, dynamic_placeholders, static_placeholders)

    # Display results
    st.write(f"**Predicted Category:** {predicted_category_name}")
    st.write("**Chatbot Response:**")
    st.markdown(response, unsafe_allow_html=True)  # Allow HTML for bold formatting
