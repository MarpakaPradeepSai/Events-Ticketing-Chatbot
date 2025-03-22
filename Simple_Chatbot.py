import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import spacy
import os
import requests
import time  # For simulating processing time
import warnings

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

# Path where you want to save the downloaded model files
model_dir = "./distilgpt2_model"

# Ensure model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to download model files from GitHub
def download_model_files(model_dir):
    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            response = requests.get(url, stream=True) # Use stream=True for potentially large files
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192): # Stream the download in chunks
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                print(f"Downloaded {filename} successfully.")
            else:
                st.error(f"Failed to download {filename} from GitHub. Status code: {response.status_code}")
                return False
    return True

# Download model files if not already present
if not all(os.path.exists(os.path.join(model_dir, file)) for file in MODEL_FILES):
    if download_model_files(model_dir):
        st.success("DistilGPT2 model files downloaded successfully!")
    else:
        st.error("Failed to download DistilGPT2 model files. Please check console for errors.")
        st.stop() # Stop execution if download fails

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

nlp = load_spacy_model()

# Load model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.stop()

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

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question):
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

# Function to generate response with DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Instruction: {instruction} Response:"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512) # Added truncation and max_length
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
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


# Streamlit UI
st.title("🎟️ Advanced Events Ticketing Chatbot")
st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input box at the bottom
if prompt := st.chat_input("Enter your question:"):
    prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

    if not prompt.strip():
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="🤖"):
            st.error("⚠️ Please enter a valid question. You cannot send empty messages.")
        st.session_state.chat_history.append({"role": "assistant", "content": "Please enter a valid question. You cannot send empty messages.", "avatar": "🤖"})
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            thinking_dots = "... Thinking..."
            message_placeholder.markdown(thinking_dots)
            time.sleep(0.5)

            dynamic_placeholders = extract_dynamic_placeholders(prompt)

            # Generate response using DistilGPT2
            bot_response = generate_response(model, tokenizer, prompt)
            response_with_placeholders = replace_placeholders(bot_response, dynamic_placeholders, static_placeholders)

            full_response = response_with_placeholders

            message_placeholder.empty()
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})

# Conditionally display reset button
if st.session_state.chat_history:
    st.markdown(
        """
        <style>
        .stButton>button {
            background: linear-gradient(90deg, #ff8a00, #e52e71);
            color: white !important;
            border: none;
            border-radius: 25px;
            padding: 10px 20px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Reset Chat", key="reset_button"):
        st.session_state.chat_history = []
        st.rerun()
