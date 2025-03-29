import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
import os
import requests
import time  # For simulating processing time

# --- Constants and Setup ---
MODEL_DIR = "./albert_model"
REPO_URL = 'https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model'
FILES_TO_DOWNLOAD = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json']
SPACY_MODEL_NAME = "en_core_web_trf" # Using transformer model for potentially better NER

CATEGORY_LABELS = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 3: "check_cancellation_fee", 4: "check_cancellation_policy",
    5: "check_privacy_policy", 6: "check_refund_policy", 7: "customer_service", 8: "delivery_options", 9: "delivery_period",
    10: "event_organizer", 11: "find_ticket", 12: "find_upcoming_events", 13: "get_refund", 14: "human_agent", 15: "information_about_tickets",
    16: "information_about_type_events", 17: "pay", 18: "payment_methods", 19: "report_payment_issue", 20: "sell_ticket", 21: "track_cancellation",
    22: "track_refund", 23: "transfer_ticket", 24: "upgrade_ticket"
}

RESPONSES = {
    'buy_ticket': "To purchase a ticket for the {{EVENT}} in {{CITY}}, please visit our website at {{WEBSITE_URL}}, navigate to the event page, select the desired ticket type, and follow the checkout process. Available payment methods include {{PAYMENT_METHODS}}.",
    'cancel_ticket': 'To cancel your ticket for the {{EVENT}} in {{CITY}}, please follow these steps:\n\n1. Access {{WEBSITE_URL}} and sign in to your account.\n2. Go to the {{CANCEL_TICKET_SECTION}} section.\n3. Locate your upcoming events and click on the {{EVENT}} in {{CITY}}.\n4. Select the {{CANCEL_TICKET_OPTION}} option.\n5. Complete the prompts to finalize your cancellation.\n\nNote that cancellations might be subject to fees as outlined in our {{CANCELLATION_POLICY_LINK}}. If any issues arise, do not hesitate to reach out to our customer support for further help.',
    'change_personal_details_on_ticket': "To change personal details on your ticket for {{EVENT}} in {{CITY}}, please log into your account on {{WEBSITE_URL}}, go to 'My Tickets', select the relevant ticket, and look for an option like 'Edit Details' or 'Manage Ticket'. If you cannot find this option, please contact {{CUSTOMER_SERVICE_CONTACT}} for assistance.",
    'check_cancellation_fee': "Cancellation fees vary depending on the event and the time of cancellation. Please refer to the specific terms and conditions for the {{EVENT}} in {{CITY}} or check our general cancellation policy here: {{CANCELLATION_POLICY_LINK}}.",
    'check_cancellation_policy': "Our cancellation policy can be found here: {{CANCELLATION_POLICY_LINK}}. It outlines the conditions under which tickets can be cancelled and any applicable fees or deadlines.",
    'check_privacy_policy': "Your privacy is important to us. You can review our full privacy policy here: {{PRIVACY_POLICY_LINK}}.",
    'check_refund_policy': "Our refund policy details the circumstances under which refunds are issued. Please review it here: {{REFUND_POLICY_LINK}}. For specific events like {{EVENT}} in {{CITY}}, refund conditions might vary.",
    'customer_service': "You can contact our customer service team through the following methods:\n- Phone: {{CUSTOMER_SERVICE_PHONE}}\n- Email: {{CUSTOMER_SERVICE_EMAIL}}\n- Live Chat on {{WEBSITE_URL}} during business hours.",
    'delivery_options': "We offer several delivery options for your tickets, including:\n- E-tickets (sent via email)\n- Mobile tickets (accessible via our app)\n- Postal delivery (where applicable, fees may apply)\nYou can usually select your preferred option during checkout.",
    'delivery_period': "E-tickets and mobile tickets are typically delivered within minutes of purchase confirmation. Postal delivery times vary based on your location, usually taking {{POSTAL_DELIVERY_ESTIMATE}} business days.",
    'event_organizer': "The organizer for {{EVENT}} in {{CITY}} is {{EVENT_ORGANIZER_NAME}}. For specific questions about the event content or venue rules, you might need to contact them directly via {{EVENT_ORGANIZER_CONTACT}}.",
    'find_ticket': "To find your purchased tickets, please log in to your account on {{WEBSITE_URL}} and navigate to the 'My Tickets' or 'Order History' section. Your tickets for {{EVENT}} in {{CITY}} should be listed there.",
    'find_upcoming_events': "You can find upcoming events, including those in {{CITY}}, by visiting the 'Events' section on {{WEBSITE_URL}}. You can filter by location, date, and category.",
    'get_refund': "To request a refund for {{EVENT}} in {{CITY}}, please check if your situation meets the criteria outlined in our refund policy ({{REFUND_POLICY_LINK}}). If eligible, log into your account, find the order, and look for a 'Request Refund' option. If unavailable, contact {{CUSTOMER_SERVICE_CONTACT}}.",
    'human_agent': "If you need further assistance, I can connect you with a human agent. Please confirm if you'd like me to transfer you.",
    'information_about_tickets': "Tickets grant access to the specified {{EVENT}} in {{CITY}}. Different ticket types (e.g., VIP, General Admission) may offer different perks or access levels. Please check the event details page on {{WEBSITE_URL}} for specifics.",
    'information_about_type_events': "We host a variety of events, including concerts, sports matches, theatre shows, festivals, and conferences. You can browse different event types on {{WEBSITE_URL}}.",
    'pay': "During checkout for {{EVENT}} in {{CITY}}, you'll be prompted to enter your payment details. We accept {{PAYMENT_METHODS}}. Follow the on-screen instructions to complete your purchase.",
    'payment_methods': "We accept the following payment methods: {{PAYMENT_METHODS}}. You can select your preferred method during the checkout process.",
    'report_payment_issue': "If you encountered a payment issue while trying to buy tickets for {{EVENT}} in {{CITY}}, please double-check your card details, ensure sufficient funds, and try again. If the problem persists, contact your bank or our {{CUSTOMER_SERVICE_CONTACT}} with details of the error.",
    'sell_ticket': "Ticket resale policies vary. Some events allow resale through authorized platforms, while others prohibit it. Please check the terms for {{EVENT}} in {{CITY}} or our general resale policy at {{RESALE_POLICY_LINK}}.",
    'track_cancellation': "To track the status of your ticket cancellation for {{EVENT}} in {{CITY}}, please log in to your account on {{WEBSITE_URL}} and check your order history or the 'My Cancellations' section. You should also receive email updates.",
    'track_refund': "You can track your refund status by logging into your account on {{WEBSITE_URL}} and checking your order details or a dedicated 'Refund Status' section. Refunds typically take {{REFUND_PROCESSING_TIME}} business days to process after approval.",
    'transfer_ticket': "Ticket transfer options depend on the event. If allowed for {{EVENT}} in {{CITY}}, you can usually transfer tickets via your account on {{WEBSITE_URL}}. Look for a 'Transfer Tickets' option within your order details. Restrictions may apply.",
    'upgrade_ticket': "To inquire about upgrading your ticket for {{EVENT}} in {{CITY}} (e.g., from General Admission to VIP), please contact {{CUSTOMER_SERVICE_CONTACT}}. Upgrades are subject to availability and price differences."
}

STATIC_PLACEHOLDERS = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{CANCEL_TICKET_SECTION}}": "'My Tickets' or 'Order History'",
    "{{CANCEL_TICKET_OPTION}}": "'Cancel Order' or 'Request Cancellation'",
    "{{CANCELLATION_POLICY_LINK}}": "www.events-ticketing.com/cancellation-policy",
    "{{PRIVACY_POLICY_LINK}}": "www.events-ticketing.com/privacy",
    "{{REFUND_POLICY_LINK}}": "www.events-ticketing.com/refund-policy",
    "{{CUSTOMER_SERVICE_CONTACT}}": "our Customer Service team",
    "{{CUSTOMER_SERVICE_PHONE}}": "1-800-EVENT-HELP",
    "{{CUSTOMER_SERVICE_EMAIL}}": "support@events-ticketing.com",
    "{{PAYMENT_METHODS}}": "Visa, MasterCard, American Express, and PayPal",
    "{{POSTAL_DELIVERY_ESTIMATE}}": "5-7",
    "{{EVENT_ORGANIZER_NAME}}": "the Event Organizer", # Default placeholder if not specified
    "{{EVENT_ORGANIZER_CONTACT}}": "their official website or contact channels", # Default placeholder
    "{{RESALE_POLICY_LINK}}": "www.events-ticketing.com/resale-policy",
    "{{REFUND_PROCESSING_TIME}}": "5-10",
}

# --- Helper Functions ---

# Function to download files from GitHub
def download_from_github(repo_url, file_name, save_path):
    # Check if file already exists
    if os.path.exists(save_path):
        # print(f"File {file_name} already exists. Skipping download.")
        return True # Indicate success (already exists)

    # If file doesn't exist, proceed with download
    raw_url = f"{repo_url.replace('github.com', 'raw.githubusercontent.com').replace('/raw/', '/')}/{file_name}"
    # print(f"Attempting to download from: {raw_url}") # Debugging print
    try:
        response = requests.get(raw_url, stream=True, timeout=30) # Added stream and timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # print(f"Successfully downloaded {file_name}") # Debugging print
        return True # Indicate success
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download {file_name} from GitHub. Error: {e}")
        # Clean up partially downloaded file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
        return False # Indicate failure

# Load the spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        # Try loading the model directly
        nlp = spacy.load(SPACY_MODEL_NAME)
        return nlp
    except OSError:
        # If not found, download it
        st.info(f"Downloading spaCy model ({SPACY_MODEL_NAME})... This may take a moment.")
        try:
            spacy.cli.download(SPACY_MODEL_NAME)
            nlp = spacy.load(SPACY_MODEL_NAME)
            st.success(f"spaCy model ({SPACY_MODEL_NAME}) downloaded and loaded successfully.")
            return nlp
        except Exception as e:
            st.error(f"Failed to download or load spaCy model ({SPACY_MODEL_NAME}): {e}")
            return None

# Load the fine-tuned model and tokenizer from the local directory
@st.cache_resource
def load_hf_model_and_tokenizer(model_directory):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        model.eval()  # Set to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer from {model_directory}: {str(e)}")
        return None, None

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp_model):
    if nlp_model is None: # Handle case where spacy model failed to load
        return {"{{EVENT}}": "the event", "{{CITY}}": "the city"}

    # Process the user question through SpaCy NER model
    doc = nlp_model(user_question)

    # Initialize dictionary to store dynamic placeholders
    dynamic_placeholders = {}
    event_found = False
    city_found = False

    # Extract entities and map them to placeholders
    # Prioritize longer matches if overlapping (spaCy often handles this well)
    for ent in doc.ents:
        # You might need to adjust these labels based on your fine-tuned spaCy model or use more robust logic
        if ent.label_ in ["EVENT", "WORK_OF_ART", "PRODUCT", "ORG"] and not event_found: # Broaden potential event labels
             event_text = ent.text.strip().title()
             # Basic filtering (optional): avoid capturing very short/generic terms unless context strongly suggests it
             if len(event_text.split()) > 1 or event_text.lower() not in ["event", "ticket", "show"]:
                 dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                 event_found = True
        elif ent.label_ == "GPE" and not city_found: # GPE is the label for geopolitical entities (cities, countries)
            city_text = ent.text.strip().title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            city_found = True

        # Stop if both are found
        if event_found and city_found:
            break

    # If no specific event or city was found, add default bolded values
    if not event_found:
        dynamic_placeholders['{{EVENT}}'] = "<b>the event</b>"
    if not city_found:
        dynamic_placeholders['{{CITY}}'] = "<b>the city</b>"

    return dynamic_placeholders

# Function to get bot response (NEW FUNCTION)
def get_bot_response(user_input, hf_model, hf_tokenizer, spacy_nlp, device):
    """Generates a response from the chatbot based on user input."""
    if not user_input or not user_input.strip():
        return "⚠️ Please enter a valid question. You cannot send empty messages."

    try:
        # Extract dynamic placeholders
        dynamic_placeholders = extract_dynamic_placeholders(user_input, spacy_nlp)

        # Tokenize input
        inputs = hf_tokenizer(user_input, padding=True, truncation=True, return_tensors="pt", max_length=512) # Added max_length
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = hf_model(**inputs)
            logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        predicted_category_index = prediction.item()
        predicted_category_name = CATEGORY_LABELS.get(predicted_category_index, "Unknown Category")

        # Get and format response
        initial_response = RESPONSES.get(predicted_category_name, "Sorry, I didn't understand your request. Can you please rephrase or ask something else?") # Improved fallback
        final_response = replace_placeholders(initial_response, dynamic_placeholders, STATIC_PLACEHOLDERS)

        return final_response

    except Exception as e:
        st.error(f"An error occurred during response generation: {e}")
        return "Sorry, I encountered an internal error. Please try again later."


# --- Main App Logic ---

st.set_page_config(page_title="Events Ticketing Chatbot", layout="wide")
st.title("🎫 Simple Events Ticketing Chatbot")
st.write("Ask me anything about ticketing for your events, or select a common question below.")

# --- Model Loading ---
# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download all model files from GitHub - with progress and error handling
all_downloads_successful = True
with st.spinner("Downloading model files (first time only)..."):
    for file in FILES_TO_DOWNLOAD:
        save_path = os.path.join(MODEL_DIR, file)
        if not download_from_github(REPO_URL, file, save_path):
            all_downloads_successful = False
            break # Stop trying if one fails

if not all_downloads_successful:
    st.error("Failed to download necessary model files. Please check the repository URL and your internet connection.")
    st.stop()

# Load models (spaCy and Hugging Face)
nlp = load_spacy_model()
model, tokenizer = load_hf_model_and_tokenizer(MODEL_DIR)

# Check if models loaded successfully
if model is None or tokenizer is None or nlp is None:
    st.error("Critical error: Failed to load necessary AI models. The application cannot proceed.")
    st.stop()

# Set device to CPU (safer for general deployment)
device = torch.device("cpu")
model.to(device)

# --- Chat Interface ---

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display reference questions select box
st.subheader("Quick Questions:")
reference_questions = [
    "How do I buy a ticket?",
    "How can I cancel my ticket for the Rock Concert in London?",
    "How can I change my personal details on the ticket?",
    "What is the cancellation fee for the Theatre Play in Paris?",
    "What is the refund policy?",
    "How can I contact customer service?",
    "What are the delivery options for tickets?",
    "How do I find my tickets for the Music Festival?",
    "How can I check upcoming events in New York?"
]
selected_question = st.selectbox(
    "Choose a reference question:",
    reference_questions,
    index=None, # Don't pre-select anything
    placeholder="Select a common question..."
)

# Button to ask the selected question
if st.button("Ask Selected Question", disabled=(selected_question is None)):
    if selected_question:
        # Capitalize first letter
        question_to_ask = selected_question[0].upper() + selected_question[1:]

        # Add user message to chat history and display it
        st.session_state.chat_history.append({"role": "user", "content": question_to_ask, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question_to_ask, unsafe_allow_html=True)

        # Get and display bot response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                bot_response = get_bot_response(question_to_ask, model, tokenizer, nlp, device)
            st.markdown(bot_response, unsafe_allow_html=True)

        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "avatar": "🤖"})

        # Clear the selection box after asking (optional)
        # st.session_state.selectbox_key = None # Requires adding a key to selectbox
        st.rerun() # Rerun to update the UI smoothly

# Display chat messages from history
st.markdown("---") # Separator
st.subheader("Chat History")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input at the bottom
if prompt := st.chat_input("Enter your question here:"):
    # Capitalize the first letter
    prompt_capitalized = prompt[0].upper() + prompt[1:] if prompt else prompt

    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt_capitalized, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_capitalized, unsafe_allow_html=True)

    # Get and display bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            bot_response = get_bot_response(prompt_capitalized, model, tokenizer, nlp, device)

        # Handle potential errors from get_bot_response which return specific strings
        if bot_response.startswith("⚠️") or bot_response.startswith("Sorry, I encountered an internal error"):
             st.warning(bot_response) # Use warning for input errors or internal issues
        else:
            st.markdown(bot_response, unsafe_allow_html=True)


    # Add bot response to chat history (even if it's an error message from the bot logic)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "avatar": "🤖"})

    # Rerun to ensure the message list is updated immediately
    st.rerun()


# Conditionally display reset button (at the bottom or sidebar)
if st.session_state.chat_history:
    st.markdown("---") # Separator
    if st.button("Reset Chat"):
        st.session_state.chat_history.clear()
        st.rerun() # Rerun to clear the displayed chat
