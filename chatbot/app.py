import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Sustainabot",
    page_icon="♻️",
    layout="centered"
)

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
        border: 4px solid #4B5320 !important; /* dark military green */
        border-radius: 4px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        outline: none !important;
        box-shadow: 0 0 8px 2px #4B5320 !important;
        -webkit-appearance: none !important;
        appearance: none !important;
    }

    /* On focus: lighter military green with stronger glow */
    .stTextInput>div>div>input:focus,
    .stTextInput>div>div>input:focus-visible {
        border: 4px solid #6B8E23 !important; /* lighter military green */
        box-shadow: 0 0 12px 4px #6B8E23 !important;
        outline: none !important;
    }

    /* Autofill and invalid inputs override */
    input:-webkit-autofill,
    input:-webkit-autofill:focus,
    input:-webkit-autofill:active,
    input:invalid,
    input:invalid:focus {
        border: 4px solid #6B8E23 !important;
        box-shadow: 0 0 12px 4px #6B8E23 !important;
        -webkit-box-shadow: 0 0 12px 4px #6B8E23 inset !important;
        background-color: #f0f2f6 !important;
        color: black !important;
        outline: none !important;
    }

    /* Remove browser red validation icons/arrows */
    input::-webkit-validation-bubble-arrow,
    input::-webkit-validation-bubble-arrow-body {
        display: none !important;
    }

    /* Chat styling */
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


if "consent_given" not in st.session_state:
    st.session_state.consent_given = None

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "initial_slots" not in st.session_state:
    st.session_state.initial_slots = {}

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
            unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_input = ""
    st.session_state.input_key = 0
    st.session_state.is_loading = False
    st.session_state.is_creating_roadmap = False  # NEW: track roadmap generation state
    # Initial bot greeting
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": st.session_state.last_input,
                "chat_history": [msg["content"] for msg in st.session_state.messages],
                "generate_roadmap": st.session_state.is_creating_roadmap  # ✅ add this
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
        st.markdown("""
            <h3>Consent Declined</h3>
            <p style="font-size: 19px;">
                No problem, we respect your decision. ✌️ <br>
                If you change your mind, just click the button below to get started.
            </p>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        col2.button("✅ Agree", on_click=lambda: setattr(st.session_state, 'consent_given', True))

if st.session_state.consent_given is True:
    if st.session_state.form_submitted is False:
        with st.form("initial_form"):
            st.markdown("""
            <p>Please provide some basic information about your business and packaging needs:</p>
            <p>If you don't have all the details right now, no worries!</p>
            """, unsafe_allow_html=True)

            main_product = st.text_input("What is your main product?")
            product_packaging = st.text_input("How is your product currently packaged?")
            packaging_material = st.text_input("What material do you use for packaging?")
            reorder_interval = st.text_input("How often do you reorder packaging material?")
            packaging_cost = st.text_input("What is the average cost per order (in EUR)?")

            packaging_provider = st.text_input("Who is your current packaging provider?")
            shipping_location = st.text_input("Where are your products shipped to?")
            production_location = st.text_input("Where are your products produced?")
            submit_button = st.form_submit_button("Start Chat")

            if submit_button:
                st.session_state.form_submitted = True
                st.session_state.initial_slots = {
                    "main_product": main_product,
                    "product_packaging": product_packaging,              
                    "packaging_material": packaging_material,
                    "packaging_reorder_interval": reorder_interval,
                    "packaging_cost_per_order": packaging_cost,          
                    "packaging_provider": packaging_provider,
                    "packaging_budget": "",                
                    "production_location": production_location,
                    "shipping_location": shipping_location,
                    "sustainability_goals": ""        
                }

                st.rerun()

    if st.session_state.form_submitted:
        # Initialisierung des Session State + automatische Begrüßung vom Bot
        # Session state initialization
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.last_input = ""
            st.session_state.input_key = 0
            # Initial bot greeting
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "message": "",
                        "chat_history": [],
                        "slots": st.session_state.initial_slots   # <-- hier
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
                            "slots": st.session_state.initial_slots   # <-- hier
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


if st.session_state.is_loading:
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": st.session_state.last_input,
                "chat_history": [msg["content"] for msg in st.session_state.messages],
                "generate_roadmap": st.session_state.is_creating_roadmap  
            }
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

    except Exception as e:
        st.error(f"Fehler bei der Kommunikation mit dem Server: {str(e)}")
        st.session_state.is_loading = False
        st.session_state.is_creating_roadmap = False
