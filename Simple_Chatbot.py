import streamlit as st
import spacy
import os

# Function to load the model (download if necessary)
@st.cache_resource
def load_model():
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        os.system("python -m spacy download en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
    return nlp

# Streamlit App
def main():
    st.title("spaCy en_core_web_trf Demo")

    # Load the model
    nlp = load_model()

    # Text input
    text = st.text_area("Enter text for analysis:", "spaCy is an advanced NLP library.")

    if st.button("Analyze"):
        if text:
            doc = nlp(text)

            # Display entities
            st.subheader("Named Entities:")
            for ent in doc.ents:
                st.write(f"{ent.text} ({ent.label_})")

            # Display POS tags
            st.subheader("Part-of-Speech Tags:")
            for token in doc:
                st.write(f"{token.text} ({token.pos_})")

            # Display Dependency Parsing
            st.subheader("Dependency Parsing:")
            for token in doc:
                st.write(f"{token.text} -- {token.dep_} --> {token.head.text}")

        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
