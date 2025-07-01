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
from transitions import Machine

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
    # "./PDFs/Flexible_Packaging_Supplementary_information.pdf",  # PDF seems to be broken
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
    # "./PDFs/SME-EnterPRIZE-White-Paper.pdf",  # PDF seems to be broken
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

class SustainabilityConsultant:
    """Main consultant class with slot-filling capabilities"""

    STATE_GREETING = "greeting"
    STATE_SLOT_FILLING = "slot_filling"
    STATE_CONSULTATION = "consultation"
    STATE_END = "end"
    STATE_QUESTION = "asking_question"


    def generate_greeting(self) -> str:
        return (
            "Hello! I'm your sustainability consultant ♻️. I help small businesses find eco-friendly packaging solutions 📦.\n\n___\n\nTo provide you with a roadmap that helps you become more sustainable and is tailored to your current business situation, I’ll ask around 10 questions. Please take a moment to read the following instructions so you know how everything works:\n\n"
            "• Settle in and answer everything thoroughly for the best results. This should take no more than ten minutes. ☕️\n\n"
            "• Once we’ve collected all the necessary information, I’ll present a summary so you can review it and let me know if anything needs correcting.\n\n"
            "• You can also ask for a summary of the collected data at any time.\n\n"
            "• If you prefer not to share certain information, just type “none.” If you don’t know the answer, simply tell me or type “idk.” It’s not a problem! 😊\n\n"
            "• If anything in the roadmap is unclear or you’d like more information, feel free to ask.\n\n"
            "• If I’m unable to understand your message, even after you’ve tried rephrasing it a few times, feel free to type “none” to skip to the next question. You’ll be able to modify your answers once the summary is shown. (Since I’m still learning, this might happen occasionally, but don’t worry, we’ll still generate a reliable roadmap based on the information I do understand. 📚)\n\n"
            "• And if you wish to end the conversation, you’re free to do so at any time.\n\n"
            "➡️ To start off, could you tell me what your business’s main product is? ✏️📋"
        )



    
    def __init__(self):
        self.state = self.STATE_GREETING

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
        
    
        self.slots = PackagingSlots()
        self.current_slot = None

        
        # Create chains
        self.slot_extractor = self.create_slot_extractor()
        self.slot_classifier = self.create_slot_classifier()
        self.question_generator = self.create_question_generator()
        self.consultation_chain = self.create_consultation_chain()
        self.goal_extractor = self.create_goal_extractor()
        self.plan_generator = self.create_implementation_plan_generator()
        self.goodbye_detector = self.goodbye_detector()  
        self.checklist_intent_detector = self.create_checklist_intent_detector()       
        self.log_writer = LogWriter()
        self.question_answerer = self.create_question_answerer()
        self.question_intent_detector = self.create_question_intent_detector()

    def create_question_answerer(self):
        """Create a chain to answer general questions about sustainability and packaging"""
        prompt = """You are a sustainability expert specializing in packaging solutions for small businesses.
        
        Answer the user's question using the provided context from sustainability documents.
        Be helpful, informative, and focus on practical advice for small businesses.
        
        If the question is not related to sustainability or packaging, politely redirect the conversation 
        back to sustainable packaging solutions.
        
        Context from knowledge base:
        {context}
        
        User question: {user_question}
        
        Answer:"""
        
        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def create_question_intent_detector(self):
        """Create a chain to detect if user is asking a general question"""
        prompt = """Analyze if the user is asking a general question about sustainability, packaging, or business practices.
        
        Look for question patterns like:
        - "What is...?", "How does...?", "Why...?", "Can you explain...?"
        - "Tell me about...", "I want to know about..."
        
        Do NOT classify as questions:
        - Statements providing information about their business
        - Greetings or goodbyes
        - Requests for checklists/roadmaps (handled separately)
        - one word 
        
        User message: {user_message}
        
        Respond with ONLY "YES" if this is a general question and you are very confident, else just "NO".
        
        Answer:"""
        
        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain

    def is_asking_question(self, user_message: str) -> bool:
        """Check if user message is asking a general question"""
        try:
            result = self.question_intent_detector.invoke({"user_message": user_message})
            return result.strip().upper() == "YES"
        except Exception as e:
            print(f"Error in question intent detection: {e}")
            # Fallback to simple keyword detection
            question_keywords = ["what is", "how does", "why", "can you explain", "tell me about", "what are", "how to"]
            return any(keyword in user_message.lower() for keyword in question_keywords) or user_message.strip().endswith('?')

    def answer_general_question(self, user_question: str, index, docs) -> tuple[str, bool, dict, None]:
        """Answer a general question using the knowledge base"""
        # Get relevant context from documents
        query_vec = get_query_embedding(user_question)
        indices, _ = search_index(index, query_vec, k=5)
        context = "\n\n".join(docs[i].page_content for i in indices)
        
        # Generate answer
        answer = self.question_answerer.invoke({
            "context": context,
            "user_question": user_question
        })
        
        is_loading = False
        log_message = {
            "user_message": user_question,
            "bot_response": answer,
            "context_used": True
        }
        
        return answer, is_loading, log_message, None  
    def should_transition_to_slot_filling(self, intent: str, extraction_result: dict) -> bool:
        """Determine if we should move from greeting to slot filling"""
        return (self.state == self.STATE_GREETING and 
                (intent in ["providing_info", "asking_question"] or extraction_result["updated_slots"]))

    def should_transition_to_consultation(self) -> bool:
        """Determine if we should move from slot filling to consultation"""
        return (self.state == self.STATE_SLOT_FILLING and self.slots.is_complete())

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

Respond with ONLY "YES" if the user really wants to end the conversation, or "NO" if they want to continue.

User message: {user_message}

Answer:"""

        chain = PromptTemplate.from_template(prompt) | self.extractor_llm | StrOutputParser()
        return chain
    def is_goodbye_message(self, user_message: str) -> bool:
        """Check if user message indicates they want to end the conversation"""
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
4. Packaging reorder interval (how often you reorder packaging, e.g., monthly, quarterly, every ... weeks)
5. Packaging cost per order (how much do you pay for the packaging per order? Prices, costs, amounts with currency, in EUR)
6. Packaging provider (who is your current supplier or provider?)
7. Packaging budget (look for budget, total amount available, spending limit)
8. Production location (in which country and city do you operate or produce? Country names, locations, "we are in", "based in")
9. Shipping location (where do you ship your product? Country names, locations)
10. Sustainability goals (do you need help with a packaging sustainability goal or want ideas?)

Rules:
- Only extract information that is explicitly mentioned
- For prices/budgets: extract numbers with currency (convert to EUR if possible, insert euro if no currency is given)
- For country: extract the specific country name
- If information is not present, respond with "NOT_FOUND"


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
    
    def create_implementation_plan_generator(self):
        prompt = """You are an expert in sustainable packaging for small businesses.

        Given the user's goal:
        {goal}

        And relevant information from our knowledge base:
        {context}

        Generate a **detailed, step-by-step checklist** (3–6 steps) to help the user implement their goal. Keep each step clear and actionable.

        Checklist:"""
        chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return chain

    def generate_goal_checklist(self, user_message: str, index, docs) -> tuple[str, bool, dict, None]:
        goal = self.goal_extractor.invoke({"user_message": user_message}).strip()
        
        if goal == "NOT_FOUND":
            response = "I couldn't identify a clear goal in your message. Could you rephrase it?"
            is_loading = False
            log_message = {"user_message": user_message, "bot_response": response}
            return response, is_loading, log_message, None

        # Get relevant context from documents
        query_vec = get_query_embedding(goal)
        indices, _ = search_index(index, query_vec, k=5)
        context = "\n\n".join(docs[i].page_content for i in indices)

        checklist = self.plan_generator.invoke({
            "goal": goal,
            "context": context
        })
        
        is_loading = False
        log_message = {"user_message": user_message, "bot_response": checklist}
        return checklist, is_loading, log_message, None

    
    def get_consultation_response(self, user_question: str, index, docs) -> str:
        """Generate consultation response using retrieved context"""

        print("\n🛠️ Roadmap is being created...\nThis might take a moment ⏳")  

        # Convert User Question to Vector
        query_vec = get_query_embedding(user_question)

        # Search for Relevant Documents (chooses the k most similar documents)
        indices, distances = search_index(index, query_vec, k=5)

        # Build Context from Documents
        context = "\n\n".join(docs[i].page_content for i in indices)

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
        consultation_response = self.consultation_chain.invoke({
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
        })

       ################################ ROADMAP ######################################################
        roadmap_prompt = f"""
            You are a sustainability expert helping a small business improve its packaging strategy.

            Business Profile:
            Main Product: {main_product}
            Production Location: {production_location}
            Shipping Location: {shipping_location}

            Packaging Details:
            Product Packaging: {product_packaging}
            Packaging Material: {packaging_material}
            Packaging Provider: {packaging_provider}
            Reorder Interval: {packaging_reorder_interval}
            Cost per Order: {packaging_cost_per_order}
            Packaging Budget: {packaging_budget}

            Sustainability Goals:
            {sustainability_goals}

            User Question:
            "{user_question}"

            Relevant Sustainability Info (from documents):
            {context}

            ---

            Write a friendly and well-structured sustainability roadmap for this business. Include:

            Thank you for providing the information about your business and packaging.

            1. 🌿 **Roadmap to Becoming a Green Thumb** – 1–2 short sentences (under this title) acknowledging their current situation and encouraging them on their journey.
            2. 💡 **Sustainability Strategy Overview** – 3–4 bullet points (no emojis) summarizing key goals.
            3. ⚡️ **Short-Term Goals (1–2 Months)** – 3–5 actionable bullet points (no emojis).
            4. 📈 **Mid-Term Goals (3–6 Months)** – 3–5 actionable bullet points (no emojis).
            5. 🌱 **Long-Term Vision (6–12 Months)** – 3–5 bullet points (no emojis).
            6. ✅ **Final Action Checklist** – A scannable to-do list (4–6 items, no emojis).

            ✍️ Style:
            - Be clear, supportive, and motivating.
            - Don't greet again when starting or giving the roadmap or say anything like "welcome".
            - Use bold text (with `**`) for section titles only — not for body text.
            - Use emojis only for section headers.
            - Keep bullet points clean and text-focused.
            - Avoid large text blocks or redundant content.
            - Use any relevant information from the imported PDFs.
            - Write directly to the business owner (use “you”).
            - Be empathetic and informative.
            - Do **not** use Markdown headers (e.g., no `#`, `##`, or `###` syntax).
            """
        
        # Generate roadmap using LLM
        roadmap_response = self.llm.invoke(roadmap_prompt).content.strip()

        # Combine consultation and roadmap with separator
        response = f"{consultation_response.strip()}\n\n---\n\n{roadmap_response}"

        #return response
        return roadmap_response, False, {"info": "Generated consultation and roadmap"}


    def generate_wrap_up_summary(self) -> str:
        """Generate a summary 4-6 sentences of the user's current situation based on filled slots."""
        slots = self.slots.slots
        prompt = (
            "Based on the user's inputs, summarize their current situation.\n"
            f"Main Product: {slots.get('main_product', '')}\n"
            f"Product Packaging: {slots.get('product_packaging', '')}\n"
            f"Packaging Material: {slots.get('packaging_material', '')}\n"
            f"Packaging Reorder Interval: {slots.get('packaging_reorder_interval', '')}\n"
            f"Packaging Cost Per Order: {slots.get('packaging_cost_per_order', '')}\n"
            f"Packaging Provider: {slots.get('packaging_provider', '')}\n"
            f"Packaging Budget: {slots.get('packaging_budget', '')}\n"
            f"Production Location: {slots.get('production_location', '')}\n"
            f"Shipping Location: {slots.get('shipping_location', '')}\n"
            f"Sustainability Goals: {slots.get('sustainability_goals', '')}\n"
            "\nCreate a short summary (4-6 sentences)."
        )
        summary = self.llm.invoke(prompt)
        return summary.content if hasattr(summary, "content") else str(summary)


    def get_response(self, user_question: str, chat_history: list, index, docs, generate_roadmap: bool = False) -> tuple[str, bool, dict, list | None]:
        """Main response generation method"""

        # Handle goodbye messages first
        if self.is_goodbye_message(user_question):
            return "Thank you for using the Sustainable Packaging Consultant! Have a green day! 🌎", False, {}, None
        
        # Classify user intent
        intent = self.slot_classifier.invoke({"user_message": user_question}).strip().lower()
        
        # Extract slots from message
        extraction_result = self.extract_slots_from_message(user_question)
        
        # Handle checklist requests (high priority)
        if self.wants_checklist(user_question):
            result = self.generate_goal_checklist(user_question, index, docs)
            # Ensure we always return 4 values
            if len(result) == 4:
                return result
            elif len(result) == 3:
                return (*result, None)
            else:
                # Fallback for unexpected result format
                return "I encountered an issue generating your checklist. Please try again.", False, {}, None
        
        # Handle general questions (but not checklists)
        if self.is_asking_question(user_question):
            result = self.answer_general_question(user_question, index, docs)
            # Ensure we always return 4 values
            if len(result) == 4:
                return result
            elif len(result) == 3:
                return (*result, None)
            else:
                return "I encountered an issue answering your question. Please try again.", False, {}, None
        
   
        if intent == "greeting":
            self.state = self.STATE_GREETING
        elif self.slots.is_complete() and self.state == self.STATE_SLOT_FILLING:
            self.state = self.STATE_CONSULTATION
        elif extraction_result["updated_slots"] and self.state == self.STATE_GREETING:
            self.state = self.STATE_SLOT_FILLING
        
        # Handle different states
        if self.state == self.STATE_GREETING:
            if intent == "providing_info" or extraction_result["updated_slots"]:
                self.state = self.STATE_SLOT_FILLING
                # Continue to slot filling logic below
            else:
                response = (
                    "Hello!👋\nI'm your sustainability consultant. I help small businesses find eco-friendly packaging solutions. "
                    f"{self.generate_slot_question()}"
                )
                log_message = {
                    "user_message": user_question,
                    "bot_response": response,
                    "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                }
                return response, False, log_message, None
        
        if self.state == self.STATE_SLOT_FILLING:
            # Update the current slot with user input
            self.update_current_slot(user_question)
            
            # Initialize response_text to avoid undefined variable
            response_text = ""
            
            if extraction_result["updated_slots"] or self.current_slot is None:
                if not self.slots.is_complete():
                    # Generate next question
                    question = self.generate_slot_question()
                    response_text = question
                    
                    log_message = {
                        "user_message": user_question,
                        "bot_response": response_text,
                        "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                    }
                    return response_text, False, log_message, None
                else:
                    # All slots complete - transition to consultation
                    self.state = self.STATE_CONSULTATION
                    if not generate_roadmap:
                        # Return loading message
                        loading_message = "🛠️ Roadmap is being created... This might take a moment ⏳"
                        log_message = {
                            "user_message": user_question,
                            "bot_response": loading_message,
                            "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                        }
                        return loading_message, True, log_message, None
                    else:
                        # Generate roadmap immediately
                        response_text, is_loading, log_data = self.get_consultation_response(user_question, index, docs)
                        self.state = self.STATE_END
                        
                        log_message = {
                            "user_message": user_question,
                            "bot_response": response_text,
                            "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                        }
                        log_message.update(log_data)
                        
                        roadmap_items = log_data.get("roadmap", []) if log_data else []
                        
                        # Add follow-up question
                        continue_message = "\n\nIs there anything unclear or do you need help with a step-by-step solution for any of the goals? Just let me know which step you need help with!"
                        response_text += continue_message
                        
                        return response_text, False, log_message, roadmap_items
            else:
                # No slots updated, ask for clarification or repeat question
                response_text = "I didn't catch that. Could you please provide more details, or type 'none' if you prefer to skip this question?"
                log_message = {
                    "user_message": user_question,
                    "bot_response": response_text,
                    "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                }
                return response_text, False, log_message, None
        
        elif self.state == self.STATE_CONSULTATION:
            if not generate_roadmap:
                # Return loading message
                loading_message = "🛠️ Roadmap is being created... This might take a moment ⏳"
                log_message = {
                    "user_message": user_question,
                    "bot_response": loading_message,
                    "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                }
                return loading_message, True, log_message, None
            else:
                # Generate roadmap
                response_text, is_loading, log_data = self.get_consultation_response(user_question, index, docs)
                self.state = self.STATE_END
                
                log_message = {
                    "user_message": user_question,
                    "bot_response": response_text,
                    "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
                }
                log_message.update(log_data)
                
                roadmap_items = log_data.get("roadmap", []) if log_data else []
                
                # Add follow-up question
                continue_message = "\n\nIs there anything unclear or do you need help with a step-by-step solution for any of the goals? Just let me know which step you need help with!"
                response_text += continue_message
                
                return response_text, False, log_message, roadmap_items
        
        elif self.state == self.STATE_END:
            # In end state, still allow checklist requests (already handled above)
            response = "Thank you for using the consultant! If you need further help, feel free to ask about specific steps or goals, or request a checklist for any particular goal."
            log_message = {
                "user_message": user_question,
                "bot_response": response
            }
            return response, False, log_message, None
        
        # Fallback for any unhandled cases
        response = "I'm not sure how to help with that. Could you please rephrase your question or let me know what specific information you need?"
        log_message = {
            "user_message": user_question,
            "bot_response": response,
            "slots": {k: v if v is not None else "" for k, v in self.slots.slots.items()}
        }
        return response, False, log_message, None
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
