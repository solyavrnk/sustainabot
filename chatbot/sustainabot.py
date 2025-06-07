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
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

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
    
    #BSR:
    #"./PDFs/Abfaelle_LEP_70x105_2021-04_barrierefrei_WEB.pdf",
    #"./PDFs/BSR_Entsorgungsbilanz_2022.pdf",
    #"./PDFs/dnk_2020_berliner_stadtreinigung_nachhaltigkeitskodex_2020.pdf",
    #"./PDFs/Infoblatt_Mieterordner_210x297_2021-04_WEB.pdf",
    
    #UBA:

    #"./PDFs/2017-03-14_texte_22-2017_bilanzierung-verpackung.pdf",
    #"./PDFs/fachbroschuere_leitfaden_fuer_umweltgerechte_versandverpackungen_im_versa.pdf",
    #"./PDFs/04_2025_texte.pdf",
    #"./PDFs/35_2025_texte_bf.pdf",
    #"./PDFs/166_2024_texte.pdf",
    #"./PDFs/texte_16-2023_texte_sustainability_key_to_stability_security_resilience_bf.pdf",
#
    ##Ellen MacArthur Foundation:
#
    #"./PDFs/The_new_plastics_economy_Rethinking_the_future_of_plastics.pdf",
    #"./PDFs/reuse_revolution_scaling_returnable_packaging_study.pdf",
    #"./PDFs/Reuse_rethinking_packaging.pdf",
    ##"./PDFs/Flexible_Packaging_Supplementary_information.pdf", -> PDF seems to be broken
    #"./PDFs/Impact_Report_Summary_2024.pdf",
    #"./PDFs/Towards_the_circular_economy.pdf",
#
    ##Others: (mainly regarding sustainability FOR small businesses)
    #"./PDFs/20171113_Small_business__big_impact_publication_ENGLISH_version.pdf",
    #"./PDFs/becoming-a-sustainable-business-apr08.pdf",
    #"./PDFs/giz2022-en-green-business-guide.pdf",
    #"./PDFs/IJHLR-Volume.pdf",
    #"./PDFs/IJSRA-2024-2500.pdf",
    #"./PDFs/LouckMartensandChoSAMPJ2010.pdf",
    #s"./PDFs/Small-Business-Britain-Small-Business-Green-Growth.pdf",
    #"./PDFs/SME-EnterPRIZE-White-Paper.pdf", -> PDF seems to be broken
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

# PS: ONLY FOR TESTING PURPOSES, I KNOW ITS NOT OUR IDEA!

# Use AcademicCloud chat model to answer
def answer_question(query: str, docs, indices):
    context = "\n\n".join(docs[i].page_content for i in indices)

    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant that supports small businesses in becoming more sustainable.\n"
        "Use the following context to answer the question as clearly and practically as possible:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    formatted_prompt = prompt_template.format(context=context, question=query)

    # Use Mistral Large on AcademicCloud
    llm = ChatOpenAI(
        base_url="https://chat-ai.academiccloud.de/v1",
        api_key=API_KEY,
        model="mistral-large-instruct",
        temperature=0.3
        #model="meta-llama-3.1-8b-instruct",
        #temperature=0.6,
    )

    return llm.invoke(formatted_prompt)

def generate_wrap_up_summary(user_profile: dict) -> str:
    wrap_up_prompt_text = wrap_up_prompt(user_profile)
    llm = ChatOpenAI(
        base_url="https://chat-ai.academiccloud.de/v1",
        api_key=API_KEY,
        model="mistral-large-instruct",
        temperature=0.3
    )
    wrap_up = llm.invoke(wrap_up_prompt_text)
    return wrap_up.content if hasattr(wrap_up, "content") else str(wrap_up)

class LogWriter:
    def __init__(self):
        self.conversation_logfile = "conversation.jsonp"
        if os.path.exists(self.conversation_logfile):
            os.remove(self.conversation_logfile)

    def make_json_safe(self, value):
        if type(value) == list:
            return [self.make_json_safe(x) for x in value]
        elif type(value) == dict:
            return {key: self.make_json_safe(value) for key, value in value.items()}
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def write(self, log_message):
        with open(self.conversation_logfile, "a") as f:
            f.write(json.dumps(self.make_json_safe(log_message), ensure_ascii=False))
            f.write("\n")

# (Slot-Filling)
def update_user_profile(current_profile: dict, slot_updates: dict) -> dict:
    updated = current_profile.copy()
    updated.update({k: v for k, v in slot_updates.items() if v})
    return updated

# Slot-extraction function (to be implemented)
def extract_slots(user_message: str) -> dict:
    slots = {}
    msg = user_message.lower()
    if "berlin" in msg:
        slots["location_city"] = "Berlin"
    return slots

def wrap_up_prompt(user_data: dict) -> str:
    prompt_template = PromptTemplate.from_template(
        '''
        Based on the user's inputs, summarize their current situation.
        Product: {product}
        Monthly/Yearly Sales Volume: {sales_volume}
        Business Location: {location_city}, {location_country}
        Shipping Destination: {shipping_destination}

        Brand Packaging: {brand_packaging}
        Shipping Packaging: {shipping_packaging}
        Packaging Material: {packaging_material}
        Supplier: {packaging_supplier}
        Units per Order: {units_per_order}
        Price per Unit: {price_per_unit}
        Order Cycle: {order_cycle}

        Packaging Budget: {packaging_budget}
        Sustainability Budget: {sustainability_budget}

        Staff Training: {staff_training}
        Customer Information on Disposal: {customer_info}
        In-Store Disposal System: {in_store_disposal}

        Business Goals / KPIs: {business_goals}

        Create a short summary (4-6 sentences).
        '''
    )
    return prompt_template.format(**{
        "product": user_data.get("product", ""),
        "sales_volume": user_data.get("sales_volume", ""),
        "location_city": user_data.get("location_city", ""),
        "location_country": user_data.get("location_country", ""),
        "shipping_destination": user_data.get("shipping_destination", ""),
        "brand_packaging": user_data.get("brand_packaging", ""),
        "shipping_packaging": user_data.get("shipping_packaging", ""),
        "packaging_material": user_data.get("packaging_material", ""),
        "packaging_supplier": user_data.get("packaging_supplier", ""),
        "units_per_order": user_data.get("units_per_order", ""),
        "price_per_unit": user_data.get("price_per_unit", ""),
        "order_cycle": user_data.get("order_cycle", ""),
        "packaging_budget": user_data.get("packaging_budget", ""),
        "sustainability_budget": user_data.get("sustainability_budget", ""),
        "staff_training": user_data.get("staff_training", ""),
        "customer_info": user_data.get("customer_info", ""),
        "in_store_disposal": user_data.get("in_store_disposal", ""),
        "business_goals": user_data.get("business_goals", "")
    })

class SustainabotAgent:
    def __init__(self):
        self.index, self.docs = load_faiss_index_and_docs()
        self.user_profile = {}  #update_user_profile
        self.wrap_up_triggered = False

    def get_response(self, user_message: str, chat_history: list[str] = []):
        slot_updates = extract_slots(user_message)
        self.user_profile = update_user_profile(self.user_profile, slot_updates)

        # Wrap-up Trigger (if user writes "summary")
        if "summary" in user_message.lower():
            summary = generate_wrap_up_summary(self.user_profile)
            self.wrap_up_triggered = True
            response_text = summary
        else:
            query_vec = get_query_embedding(user_message)
            indices, distances = search_index(self.index, query_vec, k=5)
            response = answer_question(user_message, self.docs, indices)
            response_text = response.content if hasattr(response, "content") else str(response)

        log_message = {
            "user_message": user_message,
            "bot_response": response_text,
            "slot_updates": slot_updates,
            "user_profile": self.user_profile,
            "state": {
                "wrap_up_triggered": self.wrap_up_triggered
            }
        }
        return response_text, log_message



if __name__ == "__main__":
    agent = SustainabotAgent()
    chat_history = []

    log_writer = LogWriter()
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            break
        chatbot_response, log_message = agent.get_response(user_message, chat_history)
        print("Bot: " + chatbot_response)

        chat_history.extend("User: " + user_message)
        chat_history.extend("Bot: " + chatbot_response)

        log_writer.write(log_message)