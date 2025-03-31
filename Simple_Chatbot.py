import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
import os
import requests
import time  # For simulating processing time

# --- Page Configuration and Theme ---
st.set_page_config(page_title="Event Ticketing Chatbot", page_icon="🎟️", layout="wide", initial_sidebar_state="collapsed")
st.set_theme(theme="light") # Or try "dark", "streamlit", "monokai", "agate", "oceanic"

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* --- General App Styling --- */
    body {
        background-color: #f8f9fa; /* Light background */
        color: #343a40; /* Darker text for readability */
        font-family: sans-serif;
    }
    .stApp {
        max-width: 900px; /* Limit app width for better readability on large screens */
        margin: 0 auto; /* Center the app on the page */
        padding: 20px;
    }

    /* --- Headers and Titles --- */
    h1, h2, h3, h4, h5, h6 {
        color: #e52e71; /*  Use your accent color for headers */
        font-weight: bold;
    }

    /* --- Chat Message Bubbles --- */
    .stChatMessage[data-testid="stChatMessageContent"] {
        border-radius: 20px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessageContent"].user {
        background-color: #e0f7fa; /* Light blue for user messages */
        color: #004d40; /* Dark teal text for user messages */
        border: 1px solid #b2ebf2;
    }
    .stChatMessage[data-testid="stChatMessageContent"].assistant {
        background-color: #f0f0f0; /* Light grey for assistant messages */
        color: #212529; /* Dark grey text for assistant messages */
        border: 1px solid #d9d9d9;
    }

    /* --- Input Box Styling --- */
    div[data-testid="stChatInputContainer"] > div {
        border-radius: 25px;
        border: 1px solid #ced4da; /* Light grey border */
        padding: 5px 10px;
        background-color: white;
    }

    /* --- Selectbox Styling --- */
    div[data-testid="stSelectbox"] > label {
        font-weight: bold;
        margin-bottom: 5px;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 25px;
        border: 1px solid #ced4da;
    }

    /* --- Button Styling --- */
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
        color: white !important; /* Ensure text is white */
        border: none;
        border-radius: 25px; /* Rounded corners */
        padding: 10px 20px; /* Padding */
        font-size: 1.2em; /* Font size */
        font-weight: bold; /* Bold text */
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
        display: inline-flex; /* Helps with alignment */
        align-items: center;
        justify-content: center;
        margin-top: 5px; /* Adjust slightly if needed for alignment with selectbox */
        width: auto; /* Fit content width */
        min-width: 150px; /* Optional: ensure a minimum width */
    }
    .stButton>button:hover {
        transform: scale(1.05); /* Slightly larger on hover */
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Shadow on hover */
        color: white !important; /* Ensure text stays white on hover */
    }
    .stButton>button:active {
        transform: scale(0.98); /* Slightly smaller when clicked */
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
        background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient for "Ask this question" */
        color: white !important;
    }

    /* --- Reset Button Styling --- */
    #reset_button { /* Targeting by ID, ensure your reset button has this ID if needed */
        background-color: #dc3545; /* Red color for reset */
        color: white !important;
    }
    #reset_button:hover {
        background-color: #c82333;
    }


    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar (Optional About Section) ---
with st.sidebar:
    st.header("About TicketBot")
    st.write("Your friendly AI assistant for all things event ticketing! Ask me questions about buying, cancelling, refunding tickets, and more.")
    st.markdown("[GitHub Repository](Your-GitHub-Repo-Link-Here)") # Replace with your repo link
    st.markdown("Made with ❤️ using Streamlit, Transformers, and SpaCy")


# --- Function to download files from GitHub ---
def download_from_github(repo_url, file_name, save_path):
    file_url = f"{repo_url}/{file_name}"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download {file_name} from GitHub. Status code: {response.status_code}")

# --- Paths and Model Loading ---
model_dir = "./albert_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

repo_url = 'https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model'
files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json']

for file in files:
    file_path = os.path.join(model_dir, file)
    if not os.path.exists(file_path):
        print(f"Downloading {file}...")
        try:
            download_from_github(repo_url, file, file_path)
        except Exception as e:
            st.error(f"Failed to download required model file: {file}. Error: {e}")
            st.stop()

@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.warning("SpaCy model 'en_core_web_trf' not found. Downloading...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
    return nlp

nlp = load_spacy_model()

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer from {model_dir}: {str(e)}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("Model or Tokenizer failed to load. Please check the console for errors.")
    st.stop()

device = torch.device("cpu")
model.to(device)

# --- Category Labels and Responses ---
category_labels = {
    0: "buy_ticket", 1: "cancel_ticket", 2: "change_personal_details_on_ticket", 3: "check_cancellation_fee", 4: "check_cancellation_policy",
    5: "check_privacy_policy", 6: "check_refund_policy", 7: "customer_service", 8: "delivery_options", 9: "delivery_period",
    10: "event_organizer", 11: "find_ticket", 12: "find_upcoming_events", 13: "get_refund", 14: "human_agent", 15: "information_about_tickets",
    16: "information_about_type_events", 17: "pay", 18: "payment_methods", 19: "report_payment_issue", 20: "sell_ticket", 21: "track_cancellation",
    22: "track_refund", 23: "transfer_ticket", 24: "upgrade_ticket"
}

responses = {
    'cancel_ticket': 'To cancel your ticket for the {{EVENT}} in {{CITY}}, please follow these steps:\n\n1. Access {{WEBSITE_URL}} and sign in to your account.\n2. Go to the {{CANCEL_TICKET_SECTION}} section.\n3. Locate your upcoming events and click on the {{EVENT}} in {{CITY}}.\n4. Select the {{CANCEL_TICKET_OPTION}} option.\n5. Complete the prompts to finalize your cancellation.\n\nIf any issues arise, do not hesitate to reach out to our customer support for further help.',
    'buy_ticket': "To acquire a ticket for the {{EVENT}} in {{CITY}}, please undertake the following steps:\n\n1. Access {{WEBSITE_URL}} or launch the {{APP}}.\n2. Proceed to the {{TICKET_SECTION}} segment.\n3. Input the specifics of the desired event or performance.\n4. Identify and select the event from the listed search results.\n5. Specify the quantity of tickets and choose preferred seating arrangements (if applicable).\n6. Move to the checkout phase and provide the required payment details.\n\nUpon completion of your purchase, you will receive an email confirmation containing your ticket information.",
    'change_personal_details_on_ticket': 'To update your personal details on your ticket, please adhere to the following steps:\n\n1. Go to {{WEBSITE_URL}} and sign in to your account.\n2. Proceed to the {{TICKET_SECTION}} section.\n3. Choose the specific ticket for the event in {{CITY}} that you want to amend.\n4. Click on the {{EDIT_BUTTON}} icon next to your personal details.\n5. Make the necessary amendments to your personal information.\n6. Confirm the changes by clicking {{SAVE_BUTTON}}.\n\nIf you face any issues, please contact our customer support using the contact form available on the website.',
    'check_cancellation_fee': 'To verify the cancellation fee, kindly adhere to these steps:\n\n1. Access the {{WEBSITE_URL}}.\n2. Proceed to the {{CANCELLATION_FEE_SECTION}} segment.\n3. Identify the cancellation fee information in the {{CHECK_CANCELLATION_FEE_INFORMATION}} area.\n\nIf you require additional help, do not hesitate to reach out.',
    'check_cancellation_policy': "For comprehensive details regarding our cancellation policy, please follow these instructions: \n\n1. Access our official website via the following link: {{WEBSITE_URL}}.\n2. Locate the {{CANCELLATION_POLICY_SECTION}} section within the main navigation menu.\n3. Select the {{CHECK_CANCELLATION_POLICY_OPTION}} option from the subsequent dropdown menu.\n4. Carefully read through the cancellation policy presented on the specific page.\n\nIf you require any additional information or have more inquiries, please do not hesitate to reach out to our customer support team for further assistance.",
    'check_privacy_policy': "To access our privacy policy, please adhere to the following instructions:\n\n1. Navigate to {{WEBSITE_URL}}.\n2. Scroll to the bottom section of the homepage.\n3. Select the {{PRIVACY_POLICY_LINK}} link found in the footer area.\n\nBy following these steps, you will be redirected to our comprehensive privacy policy page containing all pertinent details.",
    'check_refund_policy': "To access our refund policy, please follow the steps outlined below:\n\n1. Head over to our official site at {{WEBSITE_URL}}.\n2. Go to the {{REFUND_SECTION}} section.\n3. Select the {{REFUND_POLICY_LINK}} link.\n\nIf you require any additional help, feel free to reach out to our support team.",
    'customer_service': 'For assistance from customer service, please adhere to the following directions:\n\n1. Access our web portal at {{WEBSITE_URL}}.\n2. Proceed to the {{CUSTOMER_SERVICE_SECTION}} tab.\n3. Complete and submit the inquiry form with the details of your request.\n\nOur customer service team will address your question promptly.',
    'delivery_options': 'To view the available delivery options for your tickets, please follow these procedures:\n\n1. Navigate to our website at {{WEBSITE_URL}}.\n2. Sign into your existing account or register a new one if necessary.\n3. Access the {{DELIVERY_SECTION}} area.\n4. Choose the event for which you are buying tickets.\n5. During the purchase process, a list of delivery options will be available. Select the option that fits your requirements.\n\nFor further help, you can reach out to our customer support team using the contact information provided on our website.',
    'delivery_period': 'To find out the delivery period for your tickets, please follow these instructions carefully:\n\n1. Go to {{WEBSITE_URL}}.\n2. Log into your account with your username and password.\n3. Proceed to the {{DELIVERY_SECTION}} section.\n4. Select the order or ticket number in question to view more details.\n5. Check the delivery period listed under the {{DELIVERY_PERIOD_INFORMATION}} section.\n\nIf you need any further help, feel free to reach out to customer support via the contact options available on the website.',
    'event_organizer': 'To get in touch with an event planner, please follow these instructions:\n\n1. Access our website at {{WEBSITE_URL}}.\n2. Proceed to the {{CONTACT_SECTION}} page.\n3. Select the {{EVENT_ORGANIZER_OPTION}} feature.\n4. Complete the contact form with your information and inquiry.\n5. Submit the form by clicking the {{SEARCH_BUTTON}} button.\n\nIf you face any challenges, refer to the help section or reach out to customer support.',
    'find_ticket': 'To find your tickets for the {{EVENT}} in {{CITY}}, please undertake the following steps:\n\n1. Please access {{WEBSITE_URL}} or launch the {{APP}} application.\n2. Sign in using your established credentials.\n3. Proceed to the {{TICKET_SECTION}} area.\n4. Identify the specific event within your list of purchased tickets.\n5. Select the event to review your ticket details.\n\nFor further assistance, feel free to reach out to our support team via the Help section on our website.',
    'find_upcoming_events': 'To access details about upcoming events in {{CITY}} , follow these guidelines: \n\n1. Launch the {{APP}} or go to the {{WEBSITE_URL}}.\n2. Go to the {{EVENTS_SECTION}} section.\n3. Type the name of your city or town into the search field.\n4. Choose the type of events you are interested in, such as musical concerts, theatrical performances, or sports competitions.\n5. Hit the {{SEARCH_BUTTON}} to see a list of upcoming events.\n\nFollowing these steps should provide you with the necessary information. If you require additional help, please inform me.',
    'get_refund': "To obtain a refund for your purchased event ticket, please adhere to the following instructions:\n\n1. Access {{WEBSITE_URL}}.\n2. Sign in to your account with your username and password.\n3. Go to {{REFUND_SECTION}}.\n4. Select the specific event for which you are requesting a refund.\n5. Press {{GET_REFUND_OPTION}} to start the refund procedure.\n6. Follow the on-screen directions to finalize your refund application.\n\nUpon submission of your request, a confirmation email will be sent to you detailing the status of your refund. If you experience any difficulties, do not hesitate to reach out to our support team for further help.",
    'human_agent': 'To connect with a human agent, please adhere to the instructions provided below:\n\n1. Visit {{WEBSITE_URL}} and access your account by logging in.\n2. Find the {{CONTACT_SECTION}} area.\n3. Choose the option that allows you to speak with a human agent or representative.\n4. Proceed by following the given prompts to establish a connection with an agent.\n\nIf you face any issues along the way, please let us know so that we can offer further assistance.',
    'information_about_tickets': 'For comprehensive information regarding tickets for an event, kindly proceed with the following steps:\n\n1. Go to {{WEBSITE_URL}}.\n2. Head over to the {{EVENTS_SECTION}} section.\n3. Identify and click on the event you wish to attend.\n4. Locate and select the {{TICKETS_TAB}} tab to review all the available ticket choices and their respective prices.\n\nThis procedure will equip you with all the relevant details about the tickets for the specified event.',
    'information_about_type_events': "To explore the different types of events available, please follow the steps below:\n\n1. Go to {{WEBSITE_URL}}.\n2. Head to the {{EVENTS_SECTION}} section.\n3. Select the {{TYPE_EVENTS_OPTION}} option to access various event categories.\n4. Review the event types listed and choose the one that piques your interest.\n\nIf you need further help, feel free to request additional information.",
    'pay': "To proceed with your payment, please adhere to the following steps:\n\n1. Access {{WEBSITE_URL}}.\n2. Sign into your account with your username and password.\n3. Go to the {{PAYMENT_SECTION}} area.\n4. Choose your desired {{PAYMENT_OPTION}}.\n5. Fill in the necessary payment or transfer information.\n6. Verify the details and finalize the transaction.\n\nFor any further help, feel free to reach out to our support team via the contact information available on our website.",
    'payment_methods': "Thank you for your question regarding our payment options. Please follow these steps to review and utilize various payment methods on our website. \n\n1. Go to {{WEBSITE_URL}}.\n2. Proceed to the {{PAYMENT_SECTION}} area.\n3. Click on {{PAYMENT_OPTION}} to explore the available payment methods.\n4. Select your desired payment method and adhere to the instructions to finalize the payment.\n\nIf you need additional help, do not hesitate to reach out to our support team through the following link: {{SUPPORT_TEAM_LINK}}.",
    'report_payment_issue': "If you are experiencing difficulties with your payment, please follow the steps outlined below to report the issue:\n\n1. Navigate to our support page at {{WEBSITE_URL}}.\n2. Access the {{PAYMENT_SECTION}} section.\n3. Choose the option labeled {{PAYMENT_ISSUE_OPTION}} for reporting payment problems.\n4. Complete the form with the required information, including your payment method and any error messages you have encountered.\n5. Submit the form so our team can investigate.\n\nOur support team will contact you as soon as possible to help resolve the issue.",
    'sell_ticket': "Beginning the process of selling or exchanging your event ticket is simple. Please adhere to the following instructions:\n\n1. Go to {{WEBSITE_URL}} and enter your credentials to log in.\n2. Proceed to the {{TICKET_SECTION}} section.\n3. Identify the ticket you wish to sell or exchange.\n4. Press the {{SELL_TICKET_OPTION}} button.\n5. Fill in the necessary details and confirm your choice.\n\nBy completing these steps, you can efficiently manage your event tickets. If any complications arise, do not hesitate to seek further assistance.",
    'track_cancellation': "To verify your cancellation status, please adhere to the following steps:\n\n1. Go to {{WEBSITE_URL}}.\n2. Sign in to your account using your username and password.\n3. Proceed to the {{CANCELLATION_SECTION}} section.\n4. Click on the {{CANCELLATION_OPTION}} option to check your cancellation status.\n\nFor any additional support, do not hesitate to contact customer service.	",
    'track_refund': "To track the status of your refund, please follow these steps:\n\n1. Access our website at {{WEBSITE_URL}} and sign in to your account.\n2. Proceed to the {{REFUND_SECTION}} part within your account dashboard.\n3. Select the {{REFUND_STATUS_OPTION}} to view the present status of your refund.\n\nIf you have any additional inquiries or need more support, do not hesitate to reach out to our customer service team.	",
    'transfer_ticket': "To send your ticket for an {{EVENT}} in {{CITY}}, please adhere to these instructions:\n\n1. Access your account on {{WEBSITE_URL}}.\n2. Proceed to the {{TICKET_SECTION}} section found in your profile.\n3. Locate the specific ticket you wish to send from your listed purchases.\n4. Select the {{TRANSFER_TICKET_OPTION}} option available there.\n5. Input the recipient's email and provide any necessary details.\n6. Validate the transfer and await a confirmation email.\n\nIf you face any difficulties, refer to the help section or reach out to customer support.	",
    'upgrade_ticket': "To upgrade your ticket for the upcoming event, please follow these instructions:\n\n1. Go to {{WEBSITE_URL}}.\n2. Sign in with your username and password.\n3. Proceed to the {{TICKET_SECTION}} area.\n4. Find your current ticket purchase listed under {{UPGRADE_TICKET_INFORMATION}} and select the {{UPGRADE_TICKET_OPTION}} button.\n5. Adhere to the on-screen directions to select your intended upgrade and verify the modifications.\n\nIf you face any difficulties throughout this process, reach out to our support team for additional help."
}

static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com", "{{SUPPORT_TEAM_LINK}}": "www.support-team.com", "{{CONTACT_SUPPORT_LINK}}" : "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}" : "www.support-team.com", "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>", "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>", "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>", "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>", "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>", "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>", "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>", "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>", "{{EDIT_BUTTON}}": "<b>Edit</b>", "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>", "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>", "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>", "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>", "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>", "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>", "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>", "{{SEARCH_BUTTON}}": "<b>Search</b>", "{{SUPPORT_SECTION}}": "<b>Support</b>", "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>", "{{PAYMENT_SECTION}}": "<b>Payment</b>", "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>", "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>", "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>", "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>", "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>", "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>", "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}" : "<b>Payment</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>", "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}" : "<b>Support</b>", "{{CUSTOMER_SUPPORT_SECTION}}" : "<b>Customer Support</b>", "{{HELP_SECTION}}" : "<b>Help</b>",
    "{{TICKET_INFORMATION}}" : "<b>Ticket Information</b>", "{{UPGRADE_TICKET_BUTTON}}" : "<b>Upgrade Ticket</b>", "{{CANCEL_TICKET_BUTTON}}" : "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}" : "<b>Get Refund</b>", "{{PAYMENTS_HELP_SECTION}}" : "<b>Payments Help</b>", "{{PAYMENTS_PAGE}}" : "<b>Payments</b>",
    "{{TICKET_DETAILS}}" : "<b>Ticket Details</b>", "{{TICKET_INFORMATION_PAGE}}" : "<b>Ticket Information</b>", "{{REPORT_PAYMENT_PROBLEM}}" : "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}" : "<b>Ticket Options</b>", "{{SEND_BUTTON}}" : "<b>Send</b>", "{{PAYMENT_ISSUE_OPTION}}" : "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}" : "<b>Customer Support</b>", "{{UPGRADE_TICKET_OPTION}}" : "<b>Upgrade Ticket</b>", "{{TICKET_AVAILABILITY_TAB}}" : "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}" : "<b>Transfer Ticket</b>", "{{TICKET_MANAGEMENT}}" : "<b>Ticket Management</b>", "{{TICKET_STATUS_TAB}}" : "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}" : "<b>Ticketing</b>", "{{TICKET_TRANSFER_TAB}}" : "<b>Ticket Transfer</b>", "{{CURRENT_TICKET_DETAILS}}" : "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}" : "<b>Upgrade</b>", "{{CONNECT_WITH_ORGANIZER}}" : "<b>Connect with Organizer</b>", "{{TICKETS_TAB}}" : "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}" : "<b>Assistance Section</b>"
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    response = response.replace("{{EVENT}}", "the event") # Fallback
    response = response.replace("{{CITY}}", "the city")   # Fallback
    return response

def extract_dynamic_placeholders(user_question):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    event_found = False
    city_found = False
    for ent in doc.ents:
        if ent.label_ == "EVENT" and not event_found:
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
            event_found = True
        elif ent.label_ == "GPE" and not city_found:
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            city_found = True

    if not event_found:
        dynamic_placeholders['{{EVENT}}'] = "the event"
    if not city_found:
        dynamic_placeholders['{{CITY}}'] = "the city"
    return dynamic_placeholders


# --- Streamlit UI ---
st.header("🎟️ Event Ticketing Chatbot") # Main title using header style
st.write("Ask me anything about ticketing for your events! I can help with purchases, cancellations, refunds, and more.")

# --- Example Queries Dropdown ---
example_queries = [
    "How do I buy a ticket?",
    "What is the cancellation policy for tickets?",
    "I need to get a refund for my event ticket.",
    "Can I change the personal details on my ticket?",
    "What upcoming events are there in New York?",
    "How can I contact customer service for ticketing?",
    "Which payment methods do you accept for ticket purchases?",
    "I'm having trouble with my payment, what should I do?",
    "Is it possible to sell my ticket if I can't attend?",
    "How can I track the status of my ticket refund?",
]

selected_query = st.selectbox(
    "Choose a quick question:", # More user-friendly label for selectbox
    [""] + example_queries,
    key="query_selectbox",
    label_visibility="collapsed" # Hide default label if title is clear enough
)
process_query_button = st.button("Ask this question", key="query_button")

# --- Chat History Initialization and Display ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": "Welcome! How can I help you with event ticketing today?", "avatar": "🤖"}) # Welcome message

for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Process Selected Query from Dropdown ---
if process_query_button and selected_query:
    prompt_from_dropdown = selected_query
    prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

    st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_from_dropdown, unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        generating_response_text = "Generating response..."
        with st.spinner(generating_response_text):
            dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown)
            inputs = tokenizer(prompt_from_dropdown, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            predicted_category_name = category_labels.get(prediction, "Unknown Category")
            initial_response = responses.get(predicted_category_name, "Sorry, I didn't understand. Could you rephrase your question?") # Improved "unknown" response
            full_response = replace_placeholders(initial_response, dynamic_placeholders, static_placeholders)
            # time.sleep(1) # Simulate processing time - optional

        message_placeholder.markdown(full_response, unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})


# --- Input Box for User Questions ---
if prompt := st.chat_input("Type your question here..."): # More descriptive placeholder
    prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

    if not prompt.strip():
        st.toast("⚠️ Please enter a question.", icon="⚠️") # Using toast for empty input feedback
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_response_text = "Generating response..."
            with st.spinner(generating_response_text):
                dynamic_placeholders = extract_dynamic_placeholders(prompt)
                inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
                predicted_category_name = category_labels.get(prediction, "Unknown Category")
                initial_response = responses.get(predicted_category_name, "Sorry, I didn't understand. Could you rephrase your question?") # Improved "unknown" response
                full_response = replace_placeholders(initial_response, dynamic_placeholders, static_placeholders)
                # time.sleep(1) # Simulate processing time - optional

            message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})

# --- Reset Chat Button ---
if st.session_state.chat_history:
    if st.button("Reset Chat", key="reset_button"):
        st.session_state.chat_history = []
        st.rerun()
