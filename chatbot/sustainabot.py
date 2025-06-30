#Possible things you might need to install in your terminal before running the code:
#pip install pypdf
#pip install langchain langchain-openai langchain-huggingface langchain-community faiss-cpu numpy python-dotenv requests PyPDF2 pdfminer.six certifi

import getpass
import os
from enum import Enum
import json
from typing import List, Any, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult

from langchain_openai import ChatOpenAI
#from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

from packaging_slots import PackagingSlots
from log_writer import LogWriter  # <-- Add this import

# Other imports we need for our program (mainly for data import and handling):

# To send and get data from websites or online services (like your AI API):
import requests

# To use special tools for your computer or program (like reading command info):
import sys

# To hide or show warning messages when your program runs:
import warnings

import numpy as np

# To save your data into files and load it back later (like saving your progress):
import pickle

########## Data Binding ##########

# LangChain loader for PDFs:
from langchain_community.document_loaders import PyPDFLoader

# This will be used later to split big text into smaller pieces
# to make it easier for our LLM to process the information:
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import OpenAIEmbeddings to generate vector representations (embeddings) of text using OpenAI's API:
# (Embeddings are numerical vector representations of text that capture its semantic meaning,
# enabling efficient similarity search and retrieval in vector databases):
from langchain_community.embeddings import OpenAIEmbeddings

# Import FAISS to create and manage a high-performance vector store,
# allowing fast similarity search over large collections of embeddings.
# FAISS helps your program quickly find pieces of text that are most similar to what you ask. 
# It organizes lots of data so searching is super fast. 
# This way, your chatbot can answer questions by picking the best matching information.
from langchain_community.vectorstores import FAISS
import faiss


###################################

# the system emits a log of deprecated warnings to the console if we do not switch if off here
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable or prompt the user
'''API_KEY = os.getenv("CHAT_AI_ACCESS_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Enter your CHAT_AI_ACCESS_KEY: ")'''

API_KEY = "0cfe7441cc466c7c201a0afa04047da7" # Replace with your actual API key

########## Data Binding ########## 

print(f"Welcome to your first step towards sustainability! 🌱 I'll need a minute before we get started. 🌎")

# Your university API endpoint for embeddings and completions
BASE_URL = "https://chat-ai.academiccloud.de/v1/embeddings"

# List of our PDFs local file paths to load and process::
pdf_files = [


     # BSR:
    "./PDFs/Abfaelle_LEP_70x105_2021-04_barrierefrei_WEB.pdf",
    "./PDFs/BSR_Entsorgungsbilanz_2022.pdf",
    "./PDFs/dnk_2020_berliner_stadtreinigung_nachhaltigkeitskodex_2020.pdf",
    "./PDFs/Infoblatt_Mieterordner_210x297_2021-04_WEB.pdf",
#
    ## UBA:
    "./PDFs/2017-03-14_texte_22-2017_bilanzierung-verpackung.pdf",
    "./PDFs/fachbroschuere_leitfaden_fuer_umweltgerechte_versandverpackungen_im_versa.pdf",
    "./PDFs/04_2025_texte.pdf",
    "./PDFs/35_2025_texte_bf.pdf",
    "./PDFs/166_2024_texte.pdf",
    "./PDFs/texte_16-2023_texte_sustainability_key_to_stability_security_resilience_bf.pdf",
#
    ## Ellen MacArthur Foundation:
    "./PDFs/The_new_plastics_economy_Rethinking_the_future_of_plastics.pdf",
    "./PDFs/reuse_revolution_scaling_returnable_packaging_study.pdf",
    "./PDFs/Reuse_rethinking_packaging.pdf",
    "./PDFs/Impact_Report_Summary_2024.pdf",
    "./PDFs/Towards_the_circular_economy.pdf",
#
    ## Others: (mainly regarding sustainability FOR small businesses)
    "./PDFs/20171113_Small_business__big_impact_publication_ENGLISH_version.pdf",
    "./PDFs/becoming-a-sustainable-business-apr08.pdf",
    "./PDFs/giz2022-en-green-business-guide.pdf",
    "./PDFs/IJHLR-Volume.pdf",
    "./PDFs/IJSRA-2024-2500.pdf",
    "./PDFs/LouckMartensandChoSAMPJ2010.pdf",
    "./PDFs/Small-Business-Britain-Small-Business-Green-Growth.pdf",
    "./PDFs/Sustainability_Practices_in_Small_Business_Venture.pdf",
    ]


#Load all PDFs and convert them into LangChain documents:
all_docs = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    all_docs.extend(pages)


print(f"Total pages loaded from PDFs: {len(all_docs)}")

# Split large documents into smaller chunks (e.g., ~1000 chars):
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_split = text_splitter.split_documents(all_docs)

print(f"Total document chunks after splitting: {len(docs_split)}")

########## EMBEDDING FUNCTION ##########

# Turn each document chunk into just the text content:
texts = [doc.page_content for doc in docs_split]

# Function that sends text chunks to the embedding API and returns embeddings:
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        BASE_URL,  
        headers={
            "Authorization": f"Bearer {API_KEY}",       # Your private access key
            "Content-Type": "application/json"          # Tell the server you're sending JSON data
        },
        json={  # This is the body of your request, in JSON format
            "input": texts,                             # A list of text strings (your PDF chunks)
            "model": "e5-mistral-7b-instruct",           # The name of the embedding model to use
            "encoding_format": "float"                  # Ask for raw float vectors (not binary, etc.)
        }
    )

    # If the server says something went wrong (like wrong API key, bad input), this will show an error
    response.raise_for_status()

    # Convert the server's response from JSON into a Python dictionary
    data = response.json()

    # Extract the list of embeddings from the response
    # Each item is something like {"embedding": [0.1, 0.2, 0.3, ...]}
    return [item["embedding"] for item in data["data"]]


########## GENERATE EMBEDDINGS AND SAVE FAISS INDEX ##########

# Check if a FAISS index already exists (so we don’t redo work)
if os.path.exists("faiss_index/index.faiss"):
    print("✅ FAISS index already exists. Skipping embedding.\nAll Data is uploaded and ready for use!🌱")
else:
    try:
        # Break the text into batches to avoid sending too much at once
        batch_size = 50                       # Max number of texts per API call
        all_vectors = []                      # Will hold all the embeddings
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Total number of batches

        # Loop over each batch of texts
        for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[i:i + batch_size]   # Get a slice of texts for this batch
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")

            batch_vectors = get_embeddings(batch)  # Send texts to API to get embeddings
            all_vectors.extend(batch_vectors)      # Save the embeddings

        # Build the FAISS index
        embedding_dim = len(all_vectors[0])        # Number of dimensions in one embedding
        index = faiss.IndexFlatL2(embedding_dim)   # Use L2 (Euclidean distance) for similarity search

        vectors_np = np.array(all_vectors).astype('float32')  # Convert list to NumPy float32 array
        index.add(vectors_np)                       # Add all vectors to FAISS

        # Save the index and document chunks
        os.makedirs("faiss_index", exist_ok=True)   # Make sure folder exists
        faiss.write_index(index, "faiss_index/index.faiss")  # Save the vector index

        with open("faiss_index/docs.pkl", "wb") as f:
            pickle.dump(docs_split, f)             # Save the corresponding document chunks too

        print("✅ FAISS index and documents saved.\nAll Data is uploaded and ready for use!🌱")

    except Exception as e:
        # Print any error that happens (like API or file saving issues)
        print("❌ Error during embedding:", e)

# Load saved FAISS index and document chunks:
def load_faiss_index_and_docs():
    index = faiss.read_index("faiss_index/index.faiss")  # Load the saved vector index from file
    with open("faiss_index/docs.pkl", "rb") as f:
        docs = pickle.load(f)  # Load the document chunks that match the vectors
    return index, docs  # Return both so we can use them later


# Turn a user question (string) into a vector (embedding):
def get_query_embedding(query: str) -> np.ndarray:
    embedding = get_embeddings([query])[0]  # Call the embedding API for the query
    return np.array(embedding).astype('float32')  # Make sure it’s the right format for FAISS


# Search the FAISS index for the top k most similar chunks:
def search_index(index, query_vector: np.ndarray, k=5):
    distances, indices = index.search(np.array([query_vector]), k)  # FAISS search
    return indices[0], distances[0]  # Return best matching indices and distances

########## End of Data Binding ##########

def get_conversation_lines(): #for using the current conversation history
    if os.path.exists("conversation.jsonp"):
        with open("conversation.jsonp", encoding="utf-8") as f:
            return f.read().strip().split('\n')
    else:
        return []

class SustainabilityConsultant:
    """Main consultant class with slot-filling capabilities"""
    
    STATE_GREETING = "greeting"
    STATE_SLOT_FILLING = "slot_filling"
    STATE_CONSULTATION = "consultation"
    STATE_END = "end"

    def generate_greeting(self) -> str:
        return (
            "Hi! I’m your sustainability consultant ♻️, here to help with eco-friendly packaging 📦.\n\n"
            "I’ll generate a roadmap to help your business become more sustainable, based on a few quick questions ✏️📋.\n\n"
            "**How to answer:**\n\n"
            "• Type **none** if you prefer not to answer or if I don’t understand your input.\n\n"
            "• Type **idk** or **I don't know** if you’re unsure about an answer.\n\n"
            "• Type **summary** to see an overview of the info I’ve collected so far.\n\n"
            "• You can end the chat at any time.\n\n"
            "➡️ First question: What’s your business’s main product?"
        )
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="mistral-large-instruct",
            temperature=0.3,
            api_key=API_KEY,
            base_url="https://chat-ai.academiccloud.de/v1",
        )
        
        # Initialize slot extractor LLM (more focused for extraction)
        self.extractor_llm = ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature=0.1,
            api_key=API_KEY,
            base_url="https://chat-ai.academiccloud.de/v1",
        )
        
        self.state = self.STATE_GREETING
        self.slots = PackagingSlots()
        self.current_slot = None

        
        # Create chains
        self.slot_extractor = self.create_slot_extractor()
        self.slot_classifier = self.create_slot_classifier()
        self.question_generator = self.create_question_generator()
        self.consultation_chain = self.create_consultation_chain()
        self.goal_extractor = self.create_goal_extractor()
        #self.plan_generator = self.create_implementation_plan_generator()
        #self.roadmap_generator = self.create_roadmap_generator()
        self.goodbye_detector = self.goodbye_detector()  
        self.checklist_intent_detector = self.create_checklist_intent_detector()       
        self.log_writer = LogWriter()
    def create_checklist_intent_detector(self):
        """Creates a chain to detect if user wants a checklist, steps, or implementation plan"""
        prompt = """You are analyzing user messages to detect if they want a step-by-step checklist, implementation plan, or actionable steps.

Look for various ways people ask for structured guidance, here are few-shots:
- Direct requests: "checklist", "steps", "plan", "roadmap", "guide", "step by step"
- Action-oriented: "how do I start", "what should I do", "how to implement", "where to begin"
- Process questions: "what's the process", "walk me through", "step by step"
- Planning language: "how to plan", "implementation", "action items", "tasks"
- Sequential requests: "first step", "next steps", "in order", "sequence"

Context: This is a sustainability packaging consultant helping businesses improve their packaging.

User message: {user_message}

Respond with ONLY "YES" if the user wants a checklist/steps/plan, or "NO" if they want general consultation/advice.

Answer:"""

        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain
    
    def wants_checklist(self, user_message: str) -> bool:
        """Check if user message indicates they want a checklist or step-by-step plan"""
        try:
            result = self.checklist_intent_detector.invoke({"user_message": user_message})
            return result.strip().upper() == "YES"
        except Exception as e:
            print(f"Error in checklist intent detection: {e}")
            # Fallback to simple keyword detection
            checklist_keywords = ["checklist", "steps", "plan", "how do i start", "roadmap", "guide", "process", "implementation"]
            return any(keyword in user_message.lower() for keyword in checklist_keywords)

    def goodbye_detector(self):
        """Creates a chain to detect if user wants to end a conversation"""
        prompt = """You are detecting if a user wants to end a conversation. Analyze the user's message to determine if they are trying to say goodbye, end the conversation, or leave.

Look for various ways people say goodbye including:
- Direct goodbyes: "bye", "goodbye", "see you", "farewell"
- Casual endings: "thanks, that's all", "I'm done", "I'm good"
- Polite endings: "thank you for your help", "that's all I needed"
- Implicit endings: "gotta go", "I have to leave", "time to go"
- Appreciation + ending: "thanks for everything", "you've been helpful"
- Different languages: "auf wiedersehen", "au revoir", "ciao", "adios"

DO NOT END THE CONVERSATION  if the user is talking about how odten he reoorders pacakging:
e.q. when the user send a mensage with "once a month", "every 2 weeks", "quarterly", "monthly", "every week" or similar.

Respond with ONLY "YES" if the user wants to end the conversation, or "NO" if they want to continue.

Be generous in detecting goodbye intent - if there's any indication they want to end the conversation, respond with "YES".

User message: {user_message}

Answer:"""

        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain
    def is_goodbye_message(self, user_message: str) -> bool:
        """Check if user message indicates they want to end the conversation"""
        if any(word in user_message.lower() for word in ["summary", "summarize"]):
                return False
        try:
            result = self.goodbye_detector.invoke({"user_message": user_message})
            return result.strip().upper() == "YES"
        except Exception as e:
            print(f"Error in goodbye detection: {e}")
            # Fallback to simple keyword detection
            goodbye_keywords = ["bye", "goodbye", "quit", "exit", "thanks that's all", "gotta go"]
            return any(keyword in user_message.lower() for keyword in goodbye_keywords)
     
    def create_slot_extractor(self):
        """Creates a chain to extract slot values from user input"""
        prompt = """You are an information extractor for a sustainability consultant. Extract specific information from user messages.

Extract the following information if present:
1. Main product (what is your business's main product?)
2. Product packaging (what do you use to package one item of your product and get it ready for shipping/delivery)
3. Packaging material (which material is it? e.g., paper, organic, metal, glass, composite)
4. Packaging reorder interval (how often you reorder packaging, e.g., monthly, quarterly, every month, every week, once a month, every 2 weeks)
5. Packaging cost per order (how much do you pay for the packaging per order? Prices, costs, amounts with currency, in EUR)
6. Packaging provider (who is your current supplier or provider?)
7. Packaging budget (look for budget, total amount available, spending limit)
8. Production location (in which country and city do you operate or produce? Country names, locations, "we are in", "based in")
9. Shipping location (where do you ship your product? Country names, locations) It can be the same as the production location.
10. Sustainability goals (do you need help with a packaging sustainability goal or want ideas?)

Rules:
- Only extract information that is explicitly mentioned
- For prices/budgets: extract numbers with or without currency (convert to EUR if possible).
- For packaging reorder: extract anything that indicates frequency (e.g., "monthly", "every 2 weeks", "quarterly", "once a month")
- For country: extract the specific country name
- If information is not present, respond with "NOT_FOUND"
- Be conservative - only extract if you're confident

Format your response as JSON:
{{
    "main_product": "value or NOT_FOUND",
    "product_packaging": "value or NOT_FOUND",
    "packaging_material": "value or NOT_FOUND",
    "packaging_reorder_interval": "value or NOT_FOUND",
    "packaging_cost_per_order": "value or NOT_FOUND",
    "packaging_provider": "value or NOT_FOUND",
    "packaging_budget": "value or NOT_FOUND",
    "production_location": "value or NOT_FOUND",
    "shipping_location": "value or NOT_FOUND",
    "sustainability_goals": "value or NOT_FOUND",
}}

User message: {user_message}

Extraction:"""

        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain
    
    def create_slot_classifier(self):
        """Create a chain to classify if user is providing information or asking questions"""
        prompt = """Classify the user's intent in this conversation about sustainable packaging.

Possible classifications:
- "providing_info" - User is giving information about their packaging situation
- "asking_question" - User is asking about sustainability, alternatives, or advice
- "greeting" - User is greeting or starting conversation
- "unclear" - User's intent is unclear or they're confused

Respond with only ONE word from the above options.

User message: {user_message}
Classification:"""

        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain
    
    def create_question_generator(self):
        """Create a chain to generate questions for missing slots"""
        prompt = """You are a friendly sustainability consultant helping small businesses with packaging. 
Generate a natural question to ask for missing information.

Context: We need to gather information about the user's current packaging situation to provide personalized advice.

Current slots status:
{slots_info}

Missing information: {missing_slots}

Generate ONE friendly, conversational question to ask for the MOST IMPORTANT missing information. 
Make it sound natural and explain why you need this information.

Question:"""

        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain
    
    def create_consultation_chain(self):
        """Create a chain for providing consultation based on filled slots and retrieved context"""
        prompt = """You are an expert sustainability consultant specializing in packaging solutions for small businesses.

        User's packaging and sustainability information:
        Main Product: {main_product}
        Product Packaging: {product_packaging}
        Packaging Material: {packaging_material}
        Packaging Reorder Interval: {packaging_reorder_interval}
        Packaging Cost Per Order: {packaging_cost_per_order}
        Packaging Provider: {packaging_provider}
        Packaging Budget: {packaging_budget}
        Production Location: {production_location}
        Shipping Location: {shipping_location}
        Sustainability Goals: {sustainability_goals}

        Relevant sustainability information from knowledge base:
        {context}

        Based on this information, provide personalized, practical advice for making their packaging more sustainable. Consider:
        1. Cost-effective alternatives within their budget
        2. Country-specific regulations and options
        3. Gradual transition strategies

        Keep your advice practical, specific, and actionable. Focus on solutions that work for small businesses.

        User question: {user_question}

        Advice:"""

        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain
    
    def extract_slots_from_message(self, user_message: str) -> Dict:
        """Extract slot information from user message"""
        try:
            # Get extraction
            extraction_result = self.slot_extractor.invoke({"user_message": user_message})
            
            # Try to parse JSON
            try:
                extracted_data = json.loads(extraction_result)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract manually
                extracted_data = {
                    "main_product": "NOT_FOUND",
                    "product_packaging": "NOT_FOUND",
                    "packaging_material": "NOT_FOUND",
                    "packaging_reorder_interval": "NOT_FOUND",
                    "packaging_cost_per_order": "NOT_FOUND",
                    "packaging_provider": "NOT_FOUND",
                    "packaging_budget": "NOT_FOUND",
                    "production_location": "NOT_FOUND",
                    "shipping_location": "NOT_FOUND",
                    "sustainability_goals": "NOT_FOUND",
                }
            
            # Update slots with extracted information
            updated_slots = []
            for slot_name, value in extracted_data.items():
                if value != "NOT_FOUND" and value and slot_name in self.slots.slots:
                    self.slots.update_slot(slot_name, value)
                    updated_slots.append(slot_name)
            
            return {"updated_slots": updated_slots, "extraction": extracted_data}
            
        except Exception as e:
            print(f"Error in slot extraction: {e}")
            return {"updated_slots": [], "extraction": {}}
    
    def generate_slot_question(self) -> str:
        """Generate a question for the next missing slot"""
        missing = self.slots.get_missing_slots()
        if not missing:
            self.current_slot = None  # No missing slots to ask
            return ""
        
        # Step 1: Save the first missing slot key to self.current_slot
        self.current_slot = missing[0]
        
        slots_info = ""
        for slot, desc in PackagingSlots.REQUIRED_SLOTS.items():
            status = "✅" if self.slots.slots[slot] is not None else "❌"
            slots_info += f"{status} {desc}: {self.slots.slots[slot] or 'Not provided'}\n"
        
        # Step 2: Generate the question normally (your existing code)
        question = self.question_generator.invoke({
            "slots_info": slots_info,
            "missing_slots": [PackagingSlots.REQUIRED_SLOTS[slot] for slot in missing]
        }).strip()

        # Remove surrounding quotes if present
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1].strip()
        
        return question

    def update_current_slot(self, user_message: str):
        """Update the current slot with user input, handling 'I don't know' responses."""
        if not self.current_slot:
            return  # No slot currently being asked

        uncertain_responses = [
            "i don't know", "not sure", "no idea", "don't remember", "nope", "idk", "unknown", "none"
        ]
        normalized = user_message.strip().lower()

        if normalized in uncertain_responses or len(normalized) == 0:
            self.slots.update_slot(self.current_slot, "Unknown")
        else:
            self.slots.update_slot(self.current_slot, user_message.strip())

        self.current_slot = None  # Reset current slot after update

    def create_goal_extractor(self):
        prompt = """You are a sustainability assistant. Extract the user’s main sustainability goal from their message.

        Only extract **one clear goal**, and keep it concise (1 sentence max). If the goal is unclear, respond with "NOT_FOUND".

        User message: {user_message}

        Extracted goal:"""
        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain
    
    def get_consultation_response(self, user_question: str, index, docs) -> str:
        """Generate consultation response using retrieved context"""

        print("\n🛠️ Roadmap is being created...\nThis might take a moment ⏳")  
        # Convert User Question to Vector
        query_vec = get_query_embedding(user_question)

        # Search for Relevant Documents (chooses the k most similar documents)
        indices, distances = search_index(index, query_vec, k=3)

        # Limit the size of each document's content to speed up LLM prompt creation
        MAX_DOC_CHARS = 1000
        context = "\n\n".join(docs[i].page_content[:MAX_DOC_CHARS] for i in indices)

        # Retrieve slot values with fallback
        main_product = self.slots.slots.get("main_product") or "Not specified"
        product_packaging = self.slots.slots.get("product_packaging") or "Not specified"
        packaging_material = self.slots.slots.get("packaging_material") or "Not specified"
        packaging_reorder_interval = self.slots.slots.get("packaging_reorder_interval") or "Not specified"
        packaging_cost_per_order = self.slots.slots.get("packaging_cost_per_order") or "Not specified"
        packaging_provider = self.slots.slots.get("packaging_provider") or "Not specified"
        packaging_budget = self.slots.slots.get("packaging_budget") or "Not specified"
        production_location = self.slots.slots.get("production_location") or "Not specified"
        shipping_location = self.slots.slots.get("shipping_location") or "Not specified"
        sustainability_goals = self.slots.slots.get("sustainability_goals") or "Not specified"

        # Generate main consultation response
        '''consultation_response = self.consultation_chain.invoke({
            "context": context,
            "user_question": user_question,
            "main_product": main_product,
            "product_packaging": product_packaging,
            "packaging_material": packaging_material,
            "packaging_reorder_interval": packaging_reorder_interval,
            "packaging_cost_per_order": packaging_cost_per_order,
            "packaging_provider": packaging_provider,
            "packaging_budget": packaging_budget,
            "production_location": production_location,
            "shipping_location": shipping_location,
            "sustainability_goals": sustainability_goals,
        })'''

       ################################ ROADMAP ######################################################
        roadmap_prompt = f"""
        You are a sustainability expert helping a small business improve packaging.

        Business Profile:
        - Main Product: {main_product}
        - Production Location: {production_location}
        - Shipping Location: {shipping_location}

        Packaging Details:
        - Product Packaging: {product_packaging}
        - Packaging Material: {packaging_material}
        - Packaging Provider: {packaging_provider}
        - Reorder Interval: {packaging_reorder_interval}
        - Cost per Order: {packaging_cost_per_order}
        - Packaging Budget: {packaging_budget}

        Sustainability Goals:
        {sustainability_goals}

        User Question:
        "{user_question}"

        Relevant Info:
        {context}

        Write a clear, friendly sustainability roadmap addressed directly to the business owner.

        Include these sections with bolded titles and an emoji in each header:

        1. 🌟 Short Encouragement (1 sentence)  
        2. 🔑 Key Strategy Points (3 concise bullet points)  
        3. 📅 Goals divided into Short-term (1–3 months), Mid-term (3–6 months), and Long-term (6–12 months), with 2–3 bullet points each  
        4. ✅ Final Checklist (3–4 practical action items)  

        Avoid large blocks of text and any greetings or closing questions.  
        Use bold only for section titles.  
        Do not use markdown header syntax like ###, ##, or # anywhere.
        """
        # Generate roadmap using LLM
        roadmap_response = self.llm.invoke(roadmap_prompt).content.strip()

        #return response
        return roadmap_response, False, {"info": "Generated consultation and roadmap"}
    
    def wrap_up_prompt(self):
        """Liest die zuletzt bekannten Slot-Werte aus der conversation.jsonp-Datei und erzeugt eine Zusammenfassung."""
        try:
            with open("conversation.jsonp", "r", encoding="utf-8") as f:
                content = f.read()

            # Zerlege in mögliche JSON-Objekte anhand von Start-Zeilen
            raw_blocks = content.split('\n{')
            json_blocks = ['{' + block if not block.startswith('{') else block for block in raw_blocks]
            json_blocks = [block.strip() for block in json_blocks if block.strip()]

            slots = {}

            for block in reversed(json_blocks):
                try:
                    entry = json.loads(block)
                    if "slots" in entry and isinstance(entry["slots"], dict) and any(entry["slots"].values()):
                        slots = entry["slots"]
                        break
                except json.JSONDecodeError:
                    continue 

        except Exception as e:
                slots = self.slots.slots

        # Zusammenfassung erzeugen
        summary_parts = [
            f"📦 **Main product:** {slots.get('main_product', 'Not provided')}",
            f"🎁 **Product packaging:** {slots.get('product_packaging', 'Not provided')}",
            f"🧱 **Packaging material:** {slots.get('packaging_material', 'Not provided')}",
            f"♻️ **Reorder interval:** {slots.get('packaging_reorder_interval', 'Not provided')}",
            f"💶 **Cost per order:** {slots.get('packaging_cost_per_order', 'Not provided')}",
            f"🏭 **Packaging provider:** {slots.get('packaging_provider', 'Not provided')}",
            f"💰 **Budget:** {slots.get('packaging_budget', 'Not provided')}",
            f"🌍 **Production location:** {slots.get('production_location', 'Not provided')}",
            f"🚚 **Shipping location:** {slots.get('shipping_location', 'Not provided')}",
            f"🌱 **Sustainability goals:** {slots.get('sustainability_goals', 'Not provided')}",
        ]

        return "Here's a summary of the information so far:\n\n" + "\n".join(line + "  " for line in summary_parts)


    def get_response(self, user_question: str, chat_history: list, index, docs, generate_roadmap: bool = False) -> tuple[str, bool, dict, list | None]:

        """Main response generation method"""
        
        # Classify user intent
        intent = self.slot_classifier.invoke({"user_message": user_question}).strip().lower() #Greeting, providing info etc.

        if any(word in user_question.lower() for word in ["summary", "summarize"]):
            summary_prompt = self.wrap_up_prompt()  # Generate the summary prompt
            response = summary_prompt  # Use the generated summary prompt directly as the response
            is_loading = False
            log_message = {
                "user_message": user_question,
                "bot_response": response,
                "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
            }
            return response, is_loading, log_message, None
        
        # Extract slots from message
        extraction_result = self.extract_slots_from_message(user_question) 
        
        '''if self.wants_checklist(user_question):
            result = self.generate_goal_checklist(user_question, index, docs)
            if len(result) == 3:
                # If only 3 values returned, add None for roadmap
                return (*result, None)
            else:
                return result'''

                
        # State management
        if self.state == self.STATE_GREETING:
            if intent in ["providing_info", "asking_question"] or extraction_result["updated_slots"]:
                self.state = self.STATE_SLOT_FILLING
        
        # If all slots are filled, move to consultation
        if self.slots.is_complete():
            self.state = self.STATE_CONSULTATION
        
        # Generate response based on state
        if self.state == self.STATE_GREETING:
            response = (
                "Hello!👋\nI'm your sustainability consultant. I help small businesses find eco-friendly packaging solutions. "
                f"{self.generate_slot_question()}"
            )
            is_loading = False
            log_message = {
                "user_message": user_question,
                "bot_response": response,
                "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
            }
            return response, is_loading, log_message, None

        
        elif self.state == self.STATE_SLOT_FILLING:
           
            # First update the current slot with the user input
            self.update_current_slot(user_question)

            # Then proceed with extracting slots from the message as usual
            extraction_result = self.extract_slots_from_message(user_question)

            if extraction_result["updated_slots"]:
                ack = "Thanks for the information! "

                if not self.slots.is_complete():
                    question = self.generate_slot_question()
                    response = ack + question
                else:
                    response = ack + "Perfect! I now have all the information I need. How can I help you with sustainable packaging solutions?"
                    self.state = self.STATE_CONSULTATION
            else:
                # No new info extracted, ask for missing slots or answer question
                if intent == "asking_question":
                    response = ("I'd love to help answer your question! But first, to give you personalized advice, "
                                f"I need some information. {self.generate_slot_question()}")
                else:
                    response = self.generate_slot_question()

            is_loading = False
            log_message = {
                "user_message": user_question,
                "bot_response": response,
                "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
            }
            return response, is_loading, log_message, None

        elif self.state == self.STATE_CONSULTATION:
            if not generate_roadmap:
                # Return loading message immediately
                loading_message = "🛠️ Roadmap is being created... This might take a moment ⏳"
                is_loading = True
                log_message = {
                    "user_message": user_question,
                    "bot_response": loading_message,
                    "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                }
                return loading_message, is_loading, log_message, None
            
            else:
                # Actually generate the roadmap and return the full response
                response_text, is_loading, log_data = self.get_consultation_response(user_question, index, docs)
                self.state = self.STATE_END
                log_message = {
                    "user_message": user_question,
                    "bot_response": response_text,
                    "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                }
                log_message.update(log_data)  
                roadmap_items = log_data.get("roadmap") if log_data else []
                if roadmap_items is None:
                    roadmap_items = []
                 # Add the follow-up question to the response
            continue_message = "\n\nIs there anything unclear or do you need help with a step-by-step solution for any of the goals? Just let me know which step you need help with!"
            response_text += continue_message
            if self.wants_checklist(user_question):
                result = self.generate_goal_checklist(user_question, index, docs)
                if len(result) == 3:
                    # If only 3 values returned, add None for roadmap
                    return (*result, None)
                else:
                    return result
            return response_text, False, log_message, roadmap_items

        else:
            response = "I'm not sure how to help. Could you please rephrase your question?"
            is_loading = False
            log_message = {
                "user_message": user_question,
                "bot_response": response,
                "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
            }
            return response, is_loading, log_message, None

def main():
    """Main application loop"""

    # Greeting and short explanation of what this bot is:
    print("\n🌱 Welcome to the Sustainable Packaging Consultant! 🌎")
    print("I'll help you find eco-friendly packaging solutions for your business.\n")

    # Consent to the gathering of business info:
    print("\n🔐 Before we begin:")
    print("This chatbot will process your input to provide tailored sustainability advice.")
    print("Your input may be logged for improving the service, but no personal data is stored❕")
    consent = input("Do you agree to continue? (yes/no): ").strip().lower()

    if consent not in ["yes", "y"]:
        print("\n❌ Consent not given. Exiting the chat. Stay sustainable!🌿")
        return
    
    # Load FAISS index and documents
    try:
        index, docs = load_faiss_index_and_docs()
        print("\n✅ Knowledge base loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading knowledge base: {e}")
        return
    
    # Initialize consultant and logger
    # Bot starts the convo!
    consultant = SustainabilityConsultant()
    chat_history = []

    # Initial greeting to start the conversation
    initial_greeting, _, _ = consultant.get_response(
        user_question="hello",
        chat_history=chat_history,
        index=index,
        docs=docs
    )

    print(f"\n🤖 Consultant: {initial_greeting}")
    chat_history.append("User: hello")
    chat_history.append(f"Bot: {initial_greeting}")


    while True:
        user_message = input("\n💬 You: ")
         
        if consultant.is_goodbye_message(user_message):
            if any(word in user_message.lower() for word in ["summary", "summarize", "zusammenfassung"]):
                return False
            print("\n🌱 Thank you for using the Sustainable Packaging Consultant! Have a green day! 🌎")
            break
        
        try:
            bot_response, log_message = consultant.get_response(user_message, chat_history, index, docs)
            print(f"\n🤖 Consultant: {bot_response}")

            # Update chat history
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Bot: {bot_response}")
            
            # Show current slot status (for debugging)
            if consultant.state in [consultant.STATE_SLOT_FILLING, consultant.STATE_CONSULTATION]:
                missing = consultant.slots.get_missing_slots()
                if missing:
                    print(f"\n📋 Still need: {', '.join([PackagingSlots.REQUIRED_SLOTS[slot] for slot in missing])}")
                else:
                    print("\n✅ All information collected!")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your message.")

if __name__ == "__main__":
    main()
