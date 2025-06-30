import streamlit as st
import requests
import json
import uuid
import os
from requests.exceptions import Timeout


# Konfiguration der Seite
st.set_page_config(
    page_title="Sustainabot",
    page_icon="♻️",
    layout="centered"
)

# Get API base URL from environment variable or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost")
API_PORT = os.environ.get("API_PORT", "8001")

# Determine the correct API URL
# If API_BASE_URL is localhost, use port 8000 (internal container communication)
# Otherwise use the external port (for browser access from outside)
if "localhost" in API_BASE_URL or "127.0.0.1" in API_BASE_URL:
    API_URL = f"{API_BASE_URL}:8000/chat"
else:
    API_URL = f"{API_BASE_URL}:{API_PORT}/chat"

print(f"API_URL: {API_URL}")

st.markdown("""
<style>
    /* Streamlit UI Elemente ausblenden */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
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

# --- Consent state initialization ---
if "consent_given" not in st.session_state:
    st.session_state.consent_given = None

# --- Consent prompt flow ---
if st.session_state.consent_given is None:
    with st.container():
        st.markdown(
            """
            <h3>Before we get started:</h3>
            <p style="font-size: 19px;">
                This chatbot is designed to help you find sustainable packaging solutions ♻️.<br>
                It will process your input to provide tailored sustainability advice.<br>
                Your input may be logged for improving the service, but no personal data is stored‼️<br><br>
                🔐 <strong>Do you agree to continue?</strong>
            </p>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([5, 2, 2])
        col2.button("✅ Yes, I agree", on_click=lambda: setattr(st.session_state, "consent_given", True))
        col3.button("❌ No", on_click=lambda: setattr(st.session_state, "consent_given", False))

elif st.session_state.consent_given is False:
    with st.container():
        st.markdown(
            """
            <h3>Consent Declined</h3>
            <p style="font-size: 19px;">
                No problem, I respect your decision. ✌️ <br>
                If you change your mind, just click the button below to get started.
            </p>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([2, 1])
        col2.button("✅ Agree", on_click=lambda: setattr(st.session_state, "consent_given", True))

else:
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_input = ""
        st.session_state.input_key = 0
        st.session_state.is_loading = False
        st.session_state.is_creating_roadmap = False  # NEW: track roadmap generation state
        
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        try:
            response = requests.post(
                API_URL,
                json={
                    "message": st.session_state.last_input,
                    "chat_history": [msg["content"] for msg in st.session_state.messages],
                    "generate_roadmap": st.session_state.is_creating_roadmap,
                    "session_id": st.session_state.session_id
                },
                timeout=600  # 10 min timeout
            )
            response_data = response.json()
            st.session_state.messages.append({"role": "bot", "content": response_data["response"]})
        except Timeout:
            st.error("The server is taking too long to respond. Please try again later.")
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
                    <div>🤖 <b>Sustainabot:</b></div>
                    <div>{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)


    if st.session_state.is_loading:
        if st.session_state.is_creating_roadmap:
            st.markdown("🛠️ Roadmap is being created... This might take a moment ⏳")
        else:
            st.markdown("💬 Sustainabot is typing...")

    user_input = st.text_input("Your message:", key=f"user_input_{st.session_state.input_key}")


    if not st.session_state.is_loading:
        if user_input and user_input != st.session_state.last_input:
            st.session_state.is_loading = True
            st.session_state.is_creating_roadmap = False  
            st.session_state.last_input = user_input
            st.session_state.input_key += 1  
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.experimental_rerun()  

            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())


    if st.session_state.is_loading:
        try:
            response = requests.post(
                API_URL,
                json={
                    "message": st.session_state.last_input,
                    "chat_history": [msg["content"] for msg in st.session_state.messages],
                    "generate_roadmap": st.session_state.is_creating_roadmap,
                    "session_id": st.session_state.session_id
                },
                timeout=600  # 10 min timeout
            )

            response_data = response.json()

            # If backend signals roadmap creation, set the flag
            if response_data.get("is_loading", False):
                st.session_state.is_creating_roadmap = True
                st.experimental_rerun()
            else:
                st.session_state.is_creating_roadmap = False

                if "response" in response_data:
                    st.session_state.messages.append({"role": "bot", "content": response_data["response"]})

                roadmap_items = response_data.get("roadmap")
                if not isinstance(roadmap_items, list):
                    roadmap_items = []


                st.session_state.is_loading = False
                st.session_state.input_key += 1
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

        except Timeout:
            st.error("The server is taking too long to respond. Please try again later.")
            st.session_state.is_loading = False
            st.session_state.is_creating_roadmap = False
        except Exception as e:
            st.error(f"Error communicating with the server: {str(e)}")
            st.session_state.is_loading = False
            st.session_state.is_creating_roadmap = False

