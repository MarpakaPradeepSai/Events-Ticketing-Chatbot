import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import warnings
import spacy
import time

warnings.filterwarnings('ignore')

# GitHub directory containing the model files
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"

# List of model files to download
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    os.makedirs(model_dir, exist_ok=True)

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

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    print(f"SpaCy model loaded: {nlp}") # Debug: Check if SpaCy model is loaded
    return nlp

# Load model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Generate a chatbot response
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Instruction: {instruction} Response:"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# Define static placeholders
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

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    print(f"Inside replace_placeholders: response='{response}', dynamic_placeholders='{dynamic_placeholders}', static_placeholders='{static_placeholders}'") # Debug
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    print(f"After replace_placeholders: response='{response}'") # Debug
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    print(f"Inside extract_dynamic_placeholders: user_question='{user_question}'") # Debug
    # Process the user question through SpaCy NER model
    doc = nlp(user_question)
    print(f"SpaCy Doc Entities: {doc.ents}") # Debug: Print SpaCy entities

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

    print(f"Extracted dynamic_placeholders: {dynamic_placeholders}") # Debug
    # If no event or city was found, add default values
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"

    return dynamic_placeholders

# Streamlit UI
def main():
    st.title("🎟️ Advanced Events Ticketing Chatbot")
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None or nlp is None:
        st.error("Failed to load the model, tokenizer, or spaCy model.")
        return

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

            # Simulate bot thinking with a "typing" indicator
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""
                thinking_dots = "... Thinking..."
                message_placeholder.markdown(thinking_dots) # Show "Thinking..." initially
                time.sleep(0.5) # Small delay for visual effect

                print(f"User Prompt before NER: '{prompt}'") # Debug: Print user prompt

                # Extract dynamic placeholders
                dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)

                # Generate response from DistilGPT2
                raw_response = generate_response(model, tokenizer, prompt)
                print(f"Raw response from DistilGPT2: '{raw_response}'") # Debug: Print raw response

                # Replace placeholders in the generated response
                response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)

                full_response = response # Assign the final response

                message_placeholder.empty() # Clear "Thinking..."
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
            color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.rerun() # Rerun the Streamlit app to clear the chat display immediately

if __name__ == "__main__":
    main()
