import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
import os
import requests
import time  # For simulating processing time

# Function to download files from GitHub (same as before)
def download_from_github(repo_url, file_name, save_path):
    file_url = f"{repo_url}/{file_name}"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download {file_name} from GitHub. Status code: {response.status_code}")

# Path where you want to save the downloaded model files (same as before)
model_dir = "./albert_model"

# Ensure model directory exists (same as before)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# List of files to download from GitHub (same as before)
repo_url = 'https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model'
files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json']

# Download all model files from GitHub (same as before)
for file in files:
    download_from_github(repo_url, file, os.path.join(model_dir, file))

# Load the spaCy model for NER (same as before)
@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Initialize the spaCy model (same as before)
nlp = load_model()

# Load the fine-tuned model and tokenizer from the local directory (same as before)
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval()  # Set to evaluation mode
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

# Check if the model and tokenizer loaded successfully (same as before)
if model is None or tokenizer is None:
    st.stop()  # Halt execution if model loading fails

# Set device to CPU (Streamlit Cloud typically doesn't provide GPUs) (same as before)
device = torch.device("cpu")
model.to(device)

# Category labels mapping (same as before)
category_labels = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 3: "check_cancellation_fee", 4: "check_cancellation_policy",
    5: "check_privacy_policy", 6: "check_refund_policy", 7: "customer_service", 8: "delivery_options", 9: "delivery_period",
    10: "event_organizer", 11: "find_ticket", 12: "find_upcoming_events", 13: "get_refund", 14: "human_agent", 15: "information_about_tickets",
    16: "information_about_type_events", 17: "pay", 18: "payment_methods", 19: "report_payment_issue", 20: "sell_ticket", 21: "track_cancellation",
    22: "track_refund", 23: "transfer_ticket", 24: "upgrade_ticket"
}

# Response templates (same as before)
responses = {
    'cancel_ticket': 'To cancel your ticket for the {{EVENT}} in {{CITY}}, please follow these steps:\n\n1. Access {{WEBSITE_URL}} and sign in to your account.\n2. Go to the {{CANCEL_TICKET_SECTION}} section.\n3. Locate your upcoming events and click on the {{EVENT}} in {{CITY}}.\n4. Select the {{CANCEL_TICKET_OPTION}} option.\n5. Complete the prompts to finalize your cancellation.\n\nIf any issues arise, do not hesitate to reach out to our customer support for further help.',
    # ... (rest of the responses as provided)
}

# Define static placeholders (same as before)
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    # ... (rest of the static placeholders as provided)
}

# Function to replace placeholders (same as before)
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy (same as before)
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

# Display the reference questions box
st.subheader("You can ask the chatbot these types of questions:")
reference_questions = [
    "How do I buy a ticket?",
    "How can I cancel my ticket?",
    "How can I change my personal details on the ticket?",
    "What is the cancellation fee?",
    "What is the refund policy?",
    "How can I contact customer service?",
    "What are the delivery options for tickets?",
    "How do I find my tickets?",
    "How can I check upcoming events in my city?"
]

st.markdown(
    """
    <div style="background-color:#f0f0f0; padding: 10px; border-radius: 5px;">
    <ul style="list-style-type: none;">
    """ +
    "".join([f"<li>{q}</li>" for q in reference_questions]) +
    """
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input box at the bottom
if prompt := st.chat_input("Enter your question:"): # Renamed user_question to prompt for clarity
    # Capitalize the first letter of the user input
    prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

    # Handle empty or whitespace-only input
    if not prompt.strip():
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"}) # Still add empty input to history to show in chat
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True) # Display empty input in chat
        with st.chat_message("assistant", avatar="🤖"):
            st.error("⚠️ Please enter a valid question. You cannot send empty messages.") # Display error for empty input
        st.session_state.chat_history.append({"role": "assistant", "content": "Please enter a valid question. You cannot send empty messages.", "avatar": "🤖"}) # Add error to chat history
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        # Display user message in chat message container
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)

        # Simulate bot thinking with a "generating response..." indicator and spinner
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            generating_response_text = "Generating response..."
            # Display spinner and "Generating response..." text
            with st.spinner(generating_response_text):

                # Extract dynamic placeholders
                dynamic_placeholders = extract_dynamic_placeholders(prompt)

                # Tokenize input
                inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
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

                full_response = response # Assign the final response

            message_placeholder.empty() # Clear spinner and "Generating response..." text
            message_placeholder.markdown(full_response, unsafe_allow_html=True) # Display bot response

        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})

# Conditionally display reset button
if st.session_state.chat_history: # Check if chat_history is not empty
    st.markdown(
        """
        <style>
        .stButton>button {
            background: linear-gradient(90deg, #ff8a00, #e52e71); /* Original button gradient */
            color: white !important;
            border: none;
            border-radius: 25px; /* Original button border-radius */
            padding: 10px 20px;
            font-size: 1.2em; /* Original button font-size */
            font-weight: bold; /* Original button font-weight */
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease; /* Original button transition */
        }
        .stButton>button:hover {
            transform: scale(1.05); /* Original button hover transform */
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Original button hover box-shadow */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Reset conversation button (same as before)
    if st.button("Reset Chat"):
        st.session_state.chat_history.clear()  # Clear chat history
