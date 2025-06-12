import streamlit as st
import requests
import json

# Konfiguration der Seite
st.set_page_config(
    page_title="Sustainabot",
    page_icon="♻️",
    layout="centered"
)

# CSS für besseres Styling
st.markdown("""
<style>
    /* Hide Streamlit UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Style the input box */
    .stTextInput>div>div>input {
        background-color: #f0f2f6 !important;
        border: 1px solid #ccc !important;
        border-radius: 4px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        outline: none !important;
        -webkit-appearance: none !important;
        appearance: none !important;
        box-shadow: none !important;
    }

    /* On focus, bright green border and glow */
    .stTextInput>div>div>input:focus,
    .stTextInput>div>div>input:focus-visible {
        border: 2px solid #00FF00 !important;
        box-shadow: 0 0 8px 2px #00FF00 !important;
        outline: none !important;
    }

    /* Override all invalid styles (including autofill and autofill focus) */
    .stTextInput>div>div>input:invalid,
    .stTextInput>div>div>input:invalid:focus,
    .stTextInput>div>div>input:invalid:focus-visible,
    .stTextInput>div>div>input:-webkit-autofill,
    .stTextInput>div>div>input:-webkit-autofill:focus {
        border-color: #00FF00 !important;
        box-shadow: 0 0 8px 2px #00FF00 !important;
        outline: none !important;
        -webkit-box-shadow: 0 0 8px 2px #00FF00 inset !important;
    }

    /* Remove any red browser validation bubbles/arrows */
    .stTextInput>div>div>input::-webkit-validation-bubble-arrow,
    .stTextInput>div>div>input::-webkit-validation-bubble-arrow-body {
        display: none !important;
    }

    /* Chat styling (unchanged) */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .chat-message .avatar {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)


# Initialisierung des Session State + automatische Begrüßung vom Bot
# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_input = ""
    st.session_state.input_key = 0
    st.session_state.consent_given = None  # <-- Track consent status

    # Initial bot greeting
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": "",
                "chat_history": [],
                "consent_given": st.session_state.consent_given  # <-- send consent
            }
        )
        response_data = response.json()
        st.session_state.messages.append({"role": "bot", "content": response_data["response"]})
    except Exception as e:
        st.error(f"Fehler bei der Initialkommunikation mit dem Server: {str(e)}")

# Titel und Beschreibung
st.title("♻️ Sustainabot")
st.markdown("""
🌱 Welcome to the Sustainable Packaging Consultant! 🌎\n
I'll help you find eco-friendly packaging solutions for your business.
""")


# Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div>👤 <b>You:</b></div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <div>🤖 <b>Bot:</b></div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Zeige "generating roadmap..." Nachricht, wenn is_loading True ist
if st.session_state.get("is_loading", False):
    st.markdown("⏳ Please wait while we build your customized recommendations...")
user_input = st.text_input("Your message:", key=f"user_input_{st.session_state.input_key}")
    
if not st.session_state.get("is_loading", False):

    if user_input and user_input != st.session_state.last_input:
        # Reset loading state before sending new message
        st.session_state.is_loading = False
        # Update consent if user responds with yes or no
        lower_input = user_input.strip().lower()
        if lower_input in ["yes", "y"]:
            st.session_state.consent_given = True
        elif lower_input in ["no", "n"]:
            st.session_state.consent_given = False

        # Nachricht zum Chat-Verlauf hinzufügen
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.last_input = user_input

        
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={
                    "message": user_input,
                    "chat_history": [msg["content"] for msg in st.session_state.messages],
                    "consent_given": st.session_state.consent_given
                }
            )
            response_data = response.json()
            st.write("DEBUG: response_data =", response_data)  # <-- add this to see what you got

            st.session_state.is_loading = response_data.get("is_loading", False)

            # Now check if 'response' key exists before using it
            # Handle either a plain 'response' or a structured 'roadmap'
            if "response" in response_data:
                st.session_state.messages.append({"role": "bot", "content": response_data["response"]})
            elif "roadmap" in response_data:
                roadmap_items = response_data["roadmap"]
                # Combine roadmap items into a markdown-friendly string
                roadmap_markdown = "🗺️ **Your Roadmap to Sustainability**:\n\n"
                for i, item in enumerate(roadmap_items, 1):
                    roadmap_markdown += f"{i}. {item}\n"
                st.session_state.messages.append({"role": "bot", "content": roadmap_markdown})
            else:
                st.error("Error: Neither 'response' nor 'roadmap' found in server reply.")

            st.session_state.input_key += 1
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

        except Exception as e:
            st.error(f"Fehler bei der Kommunikation mit dem Server: {str(e)}")
