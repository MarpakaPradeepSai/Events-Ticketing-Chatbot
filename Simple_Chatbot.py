import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import spacy
import os
import requests
import time

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    os.makedirs(model_dir, exist_ok=True)

    GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
    MODEL_FILES = [
        "config.json", "generation_config.json", "merges.txt", "model.safetensors", "special_tokens_map.json", 
        "tokenizer_config.json", "vocab.json"
    ]

    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
            else:
                st.error(f"Failed to download {filename} from GitHub.")
                return False
    return True

# Load DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Load the spaCy model for NER
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Function to replace placeholders (static + dynamic placeholders)
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using spaCy NER model
def extract_dynamic_placeholders(user_question):
    # Process the user question through SpaCy NER model
    doc = nlp(user_question)
    dynamic_placeholders = {}

    # Extract entities and map them to placeholders
    for ent in doc.ents:
        if ent.label_ == "EVENT":  # Assuming 'EVENT' is the label for event names
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

# Static placeholders
static_placeholders = {
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{PAYMENT_SECTION}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>"
}

# Streamlit UI
st.title("Simple Events Ticketing Chatbot")
st.write("Ask me anything about ticketing for your events!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input box at the bottom
if prompt := st.chat_input("Enter your question:"):
    # Capitalize the first letter of the user input
    prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

    # Handle empty or whitespace-only input
    if not prompt.strip():
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="🤖"):
            st.error("⚠️ Please enter a valid question. You cannot send empty messages.")
        st.session_state.chat_history.append({"role": "assistant", "content": "Please enter a valid question. You cannot send empty messages.", "avatar": "🤖"})
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        # Display user message in chat message container
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)

        # Simulate bot thinking with a "typing" indicator
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            thinking_dots = "... Thinking..."
            message_placeholder.markdown(thinking_dots)  # Show "Thinking..." initially
            time.sleep(0.5)  # Small delay for visual effect

            # Extract dynamic placeholders
            dynamic_placeholders = extract_dynamic_placeholders(prompt)

            # Load the DistilGPT2 model and tokenizer
            model, tokenizer = load_model_and_tokenizer()
            if model is None or tokenizer is None:
                st.error("Error loading the model!")
                st.stop()

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

            # Move input tensors to device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate a response using the model
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=256,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_response = replace_placeholders(generated_text, dynamic_placeholders, static_placeholders)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

            # Update chat history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
