# config.py
import os
from dotenv import load_dotenv
# from chromadb.config import Settings as ChromaSettings # <-- REMOVE OR COMMENT OUT this import

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Configuration ---
# Recommend using Langchain's Groq integration if possible
# LLM_PROVIDER = "groq" # or "requests" if using raw requests
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-32b") # Reverted to a known good default
# LLM_MODEL_NAME = "qwen-qwq-32b" # Your original choice via requests
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" # Needed if using raw requests

# --- Vision Model Configuration ---
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "mistral-small-latest")

# --- Embedding Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# --- API Embedding Configuration ---
USE_API_EMBEDDINGS = os.getenv("USE_API_EMBEDDINGS", "true").lower() == "true"
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://sabrinekh-embedder-model.hf.space/embed")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1024))  # Default to 1024 for BAAI/bge-m3
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 5))  # Reduced default for large files
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", 120))  # Increased timeout for large files
EMBEDDING_MAX_TEXT_LENGTH = int(os.getenv("EMBEDDING_MAX_TEXT_LENGTH", 30000))  # Max characters per text

# --- Vector Store Configuration ---
# Define the persistence directory (can be None for in-memory)
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db_prod") # Use consistent variable name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_qa_prod_collection") # Use the name expected by vector_store.py

# *** Calculate the is_persistent flag ***
is_persistent = bool(CHROMA_PERSIST_DIRECTORY) # True if directory is set, False otherwise

# --- Text Splitting Configuration ---
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", (1000)))  # Restored
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # Restored

# --- Retriever Configuration ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 8)) # Renamed from RETRIEVER_SEARCH_K
VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 0.7)) # Add similarity threshold

# --- LLM Request Configuration ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0)) # Adjusted default
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 8192))

# --- Logging ---
# LOG_LEVEL = "INFO" # Can be set via environment if needed

# --- Validation ---
if not GROQ_API_KEY:
    # In a real app, might raise specific error or handle differently
    print("Warning: GROQ_API_KEY not found in environment variables.")

# --- Simplified CHROMA_SETTINGS attribute for app.py check ---
# Define a simple object or dictionary that app.py can check for is_persistent
class SimpleChromaSettings:
    # Simple placeholder class
    def __init__(self, persistent_flag):
        self.is_persistent = persistent_flag

# *** Instantiate the SimpleChromaSettings using the calculated flag ***
CHROMA_SETTINGS = SimpleChromaSettings(is_persistent) # Pass the calculated boolean heremedium