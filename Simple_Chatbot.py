import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy

# --- Load Models (Do this only once) ---
@st.cache_resource  # Cache the model to avoid reloading on each interaction
def load_models():
    """Loads the fine-tuned model, tokenizer, and SpaCy model."""
    # Path to the fine-tuned model (adjust if needed)
    path = "https://github.com/MarpakaPradeepSai/Simple-Events-Ticketing-Customer-Support-Chatbot/raw/main/ALBERT_Model"  # Important: Use /app/ for GitHub-deployed apps

    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    nlp = spacy.load("en_core_web_trf")  # Transformer-based SpaCy model

    model.eval()
    model.to("cpu")  # Use CPU in Streamlit environments

    return model, tokenizer, nlp

model, tokenizer, nlp = load_models()

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

    'buy_ticket': "To acquire a ticket for the {{EVENT}} in {{CITY}}, please follow these steps:\n\n1. Access {{WEBSITE_URL}} or launch the {{APP}}.\n2. Proceed to the {{TICKET_SECTION}} segment.\n3. Input the specifics of the desired event or performance.\n4. Identify and select the event from the listed search results.\n5. Specify the quantity of tickets and choose preferred seating arrangements (if applicable).\n6. Move to the checkout phase and provide the required payment details.\n\nUpon completion of your purchase, you will receive an email confirmation containing your ticket information.",

    'payment_methods': "Thank you for your question regarding our payment options. Please follow these steps to review and utilize various payment methods on our website. \n\n1. Go to {{WEBSITE_URL}}.\n2. Proceed to the {{PAYMENT_SECTION}} area.\n3. Click on {{PAYMENT_OPTION}} to explore the available payment methods.\n4. Select your desired payment method and adhere to the instructions to finalize the payment.\n\nIf you need additional help, do not hesitate to reach out to our support team through the following link: {{SUPPORT_TEAM_LINK}}.",

    'report_payment_issue': "If you are experiencing difficulties with your payment, please follow the steps outlined below to report the issue:\n\n1. Navigate to our support page at {{WEBSITE_URL}}.\n2. Access the {{PAYMENT_SECTION}} section.\n3. Choose the option labeled {{PAYMENT_ISSUE_OPTION}} for reporting payment problems.\n4. Complete the form with the required information, including your payment method and any error messages you have encountered.\n5. Submit the form so our team can investigate.\n\nOur support team will contact you as soon as possible to help resolve the issue.",

    'sell_ticket': "Beginning the process of selling or exchanging your event ticket is simple. Please adhere to the following instructions:\n\n1. Go to {{WEBSITE_URL}} and enter your credentials to log in.\n2. Proceed to the {{TICKET_SECTION}} area.\n3. Identify the ticket you wish to sell or exchange.\n4. Press the {{SELL_TICKET_OPTION}} button.\n5. Fill in the necessary details and confirm your choice.\n\nBy completing these steps, you can efficiently manage your event tickets. If any complications arise, do not hesitate to seek further assistance.",

    'track_cancellation': "To verify your cancellation status, please adhere to the following steps:\n\n1. Go to {{WEBSITE_URL}}.\n2. Sign in to your account using your username and password.\n3. Proceed to the {{CANCELLATION_SECTION}} section.\n4. Click on the {{CANCELLATION_OPTION}} option to check your cancellation status.\n\nFor any additional support, do not hesitate to contact customer service.	",

    'track_refund': "To track the status of your refund, please follow these steps:\n\n1. Access our website at {{WEBSITE_URL}} and sign in to your account.\n2. Proceed to the {{REFUND_SECTION}} part within your account dashboard.\n3. Select the {{REFUND_STATUS_OPTION}} to view the present status of your refund.\n\nIf you have any additional inquiries or need more support, do not hesitate to reach out to our customer service team.	",

    'transfer_ticket': "To send your ticket for an {{EVENT}} in {{CITY}}, please adhere to these instructions:\n\n1. Access your account on {{WEBSITE_URL}}.\n2. Proceed to the {{TICKET_SECTION}} section found in your profile.\n3. Locate the specific ticket you wish to send from your listed purchases.\n4. Select the {{TRANSFER_TICKET_OPTION}} option available there.\n5. Input the recipient's email and provide any necessary details.\n6. Validate the transfer and await a confirmation email.\n\nIf you face any difficulties, refer to the help section or reach out to customer support.	",

    'upgrade_ticket': "To upgrade your ticket for the upcoming event, please follow these instructions:\n\n1. Go to the {{WEBSITE_URL}}.\n2. Sign in with your username and password.\n3. Proceed to the {{TICKET_SECTION}} area.\n4. Find your current ticket purchase listed under {{UPGRADE_TICKET_INFORMATION}} and select the {{UPGRADE_TICKET_OPTION}} button.\n5. Adhere to the on-screen directions to select your intended upgrade and verify the modifications.\n\nIf you face any difficulties throughout this process, reach out to our support team for additional help.	"
}

static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}" : "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}" : "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
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
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}" : "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}" : "<b>Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}" : "<b>Customer Support</b>",
    "{{HELP_SECTION}}" : "<b>Help</b>",
    "{{TICKET_INFORMATION}}" : "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}" : "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}" : "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}" : "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}" : "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}" : "<b>Payments</b>",
    "{{TICKET_DETAILS}}" : "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}" : "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}" : "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}" : "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}" : "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}" : "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}" : "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}" : "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}" : "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}" : "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}" : "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}" : "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}" : "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}" : "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}" : "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}" : "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}" : "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}" : "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}" : "<b>Assistance Section</b>"

}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    # Replace both static and dynamic placeholders
    # First replace static placeholders
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)

    # Then replace dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)

    return response

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

# --- Streamlit UI ---
st.title("Ticketing Chatbot")

user_question = st.text_input("Enter your question:", "")

if user_question:
    dynamic_placeholders = extract_dynamic_placeholders(user_question)

    inputs = tokenizer(user_question, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to("cpu") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=-1)
    predicted_category_index = prediction.item()
    predicted_category_name = category_labels.get(predicted_category_index, "Unknown Category")

    initial_response = responses.get(predicted_category_name, "Sorry, I didn't understand your request. Please try again.")
    response = replace_placeholders(initial_response, dynamic_placeholders, static_placeholders)

    st.write(f"**Question:** {user_question}")
    st.write(f"**Predicted Category:** {predicted_category_name}")
    st.write("ChatBot Response:")
    st.write(response)
