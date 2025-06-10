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

# Initialisierung des Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Titel und Beschreibung
st.title("♻️ Sustainabot")
st.markdown("""
🌱 Welcome to the Sustainable Packaging Consultant! 🌱🌎"\n
I'll help you find eco-friendly packaging solutions for your business.
""")


# Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div>👤 <b>Du:</b></div>
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

# Eingabefeld in einem Container
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_input("Your message:", key=f"user_input_{st.session_state.input_key}")
    st.markdown('</div>', unsafe_allow_html=True)

if user_input and user_input != st.session_state.last_input:
    # Nachricht zum Chat-Verlauf hinzufügen
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_input = user_input
    
    # API-Anfrage senden
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": user_input,
                "chat_history": [msg["content"] for msg in st.session_state.messages]
            }
        )
        response_data = response.json()
        
        # Bot-Antwort zum Chat-Verlauf hinzufügen
        st.session_state.messages.append({"role": "bot", "content": response_data["response"]})
        
        # Eingabefeld leeren durch Erhöhung des Keys
        st.session_state.input_key += 1
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Fehler bei der Kommunikation mit dem Server: {str(e)}")