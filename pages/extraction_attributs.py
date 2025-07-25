import streamlit as st
st.set_page_config(page_title="PDF Attribute Extraction", layout="wide")

from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.debug("TEST DEBUG LOG: If you see this, DEBUG logging is working.")

# --- Force python to use pysqlite3 based on chromadb docs ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# --- DEBUG LOGGER INTEGRATION ---
from debug_logger import debug_logger, DebugTimer, log_streamlit_state, log_json_parsing
debug_logger.info("Extraction page loaded", context={"page": "extraction_attributs"})

import os
import time
from loguru import logger
import json
import pandas as pd
import asyncio
import subprocess
import nest_asyncio
from typing import List
import re
from groq import Groq
import requests
from sentence_transformers import SentenceTransformer
import config

nest_asyncio.apply()

# Log initial session state
debug_logger.info("Initial session state", data=dict(st.session_state), context={"page": "extraction_attributs"})

# --- UI Setup ---
st.markdown(
    """<style>
    [data-testid='stSidebarNav'] {display: none;}
    
    /* Blue band header styling */
    .header-band {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4a90e2 100%);
        color: white;
        padding: 0.1rem 0;
        margin: 0 0 0.2rem 0;
        text-align: center;
        box-shadow: 0 2px 6px rgba(30, 60, 114, 0.15);
    }
    
    .header-band h1 {
        font-size: 2em;
        margin: 0;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .header-band h2 {
        font-size: 1.2em;
        margin: 0.1rem 0 0 0;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Button styling with blue theme */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(30, 60, 114, 0.08);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #4a90e2 100%);
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(30, 60, 114, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Section headers styling */
    .section-header {
        color: #1e3c72;
        font-size: 1.2em;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }
    
    /* Info boxes styling */
    .stAlert {
        border-left: 2px solid #1e3c72;
        padding: 0.1rem 0.2rem;
    }
    
    /* Success messages styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #1e3c72;
        padding: 0.1rem 0.2rem;
    }
    
    /* Warning messages styling */
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #1e3c72;
        padding: 0.1rem 0.2rem;
    }
    
    /* Horizontal table styling */
    .horizontal-table {
        display: flex;
        flex-wrap: wrap;
        gap: 0.2rem;
        margin: 0.2rem 0;
    }
    
    .attribute-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1.5px solid #1e3c72;
        border-radius: 4px;
        padding: 0.1rem 0.2rem 0.2rem 0.2rem;
        min-width: 0;
        width: 100%;
        box-shadow: 0 1px 3px rgba(30, 60, 114, 0.04);
        margin: 0 0 0.2rem 0;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    .attribute-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(30, 60, 114, 0.08);
    }
    
    .attribute-card h4 {
        color: #1e3c72;
        margin: 0 0 0.1rem 0;
        font-size: 1em;
        font-weight: 700;
        border-bottom: 1px solid #1e3c72;
        padding-bottom: 0.1rem;
        width: 100%;
    }
    
    .attribute-value {
        background: #fff;
        border: 1px solid #dee2e6;
        border-radius: 3px;
        padding: 0.1rem 0.2rem;
        margin: 0.05rem 0 0 0;
        font-weight: 500;
        font-size: 0.95em;
        width: 100%;
        box-sizing: border-box;
    }
    
    .attribute-source {
        font-size: 0.7em;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.1rem;
    }
    
    .success-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.2rem;
    }
    
    .success-true {
        background-color: #28a745;
    }
    
    .success-false {
        background-color: #dc3545;
    }
    
    /* Data editor styling */
    .stDataFrame {
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(30, 60, 114, 0.05);
        padding: 0.1rem;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1.5px solid #1e3c72;
        border-radius: 4px;
        padding: 0.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(30, 60, 114, 0.05);
    }
    
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #1e3c72;
        margin: 0.1rem 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.8em;
        margin-bottom: 0.1rem;
    }
    
    /* Right pane styling */
    .right-pane {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 2px solid #1e3c72;
        border-radius: 0 8px 8px 0;
        padding: 0.2rem;
        box-shadow: -2px 0 6px rgba(30, 60, 114, 0.04);
        max-height: 90vh;
        overflow-y: auto;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 300px;
        overflow-y: auto;
        padding: 0.2rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(30, 60, 114, 0.04);
        margin-bottom: 0.2rem;
    }
    
    /* Chatbot styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
        margin-bottom: 1rem;
    }
    
    .chat-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1e3c72;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #1e3c72;
        margin-right: 2rem;
    }
    
    .extraction-results {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
        margin-bottom: 1rem;
    }
    
    .result-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .result-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.2);
    }
    
    .result-label {
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.5rem;
        font-weight: 500;
    }
    </style>""",
    unsafe_allow_html=True
)

# --- Chatbot Functions ---

# --- Navigation Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
    if st.button("🏠 Home"):
        st.switch_page("app.py")
    if st.button("💬 Chat with Leoparts"):
        st.switch_page("pages/chatbot.py")
    if st.button("📄 Extract a new Part"):
        st.switch_page("pages/extraction_attributs.py")
    if st.button("🔍 Debug Interface"):
        st.switch_page("debug_interface.py")
    if st.button("📊 Debug Summary"):
        st.switch_page("debug_summary.py")

def extract_json_from_string(s):
    """
    Extracts the first valid JSON object from a string.
    Returns the parsed dict, or None if not found.
    """
    if not s or not isinstance(s, str):
        return None
    # Remove <think>...</think> blocks
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    # Find the first {...} block
    match = re.search(r'\{.*\}', s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

# --- Playwright Browser Installation ---
def install_playwright_browsers():
    """Install Playwright browsers if needed."""
    logger.info("Checking and installing Playwright browsers if needed...")
    try:
        process = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                                 capture_output=True, text=True, check=False)
        if process.returncode == 0:
             logger.success("Playwright browsers installed successfully (or already exist).")
        else:
             logger.error(f"Playwright browser install command failed with code {process.returncode}.")
             logger.error(f"stdout: {process.stdout}")
             logger.error(f"stderr: {process.stderr}")
    except FileNotFoundError:
        logger.error("Could not find 'playwright' command. Is playwright installed correctly?")
        st.error("Playwright not found. Please ensure 'playwright' is in requirements.txt")
    except Exception as e:
        logger.error(f"An error occurred during Playwright browser installation: {e}", exc_info=True)
        st.warning(f"An error occurred installing Playwright browsers: {e}. Web scraping may fail.")

if 'playwright_installed' not in st.session_state:
    install_playwright_browsers()
    st.session_state.playwright_installed = True

# --- Imports ---
import config
from pdf_processor import process_uploaded_pdfs, fetch_chunks
from vector_store import (
    get_embedding_function,
    setup_vector_store
)
from llm_interface import (
    initialize_llm,
    create_pdf_extraction_chain,
    create_web_extraction_chain,
    _invoke_chain_and_process,
    scrape_website_table_html,
    create_numind_extraction_chain,
    extract_with_numind_from_bytes,
    extract_with_numind_using_schema,
    get_default_extraction_schema,
    extract_specific_attribute_from_numind_result
)
from numind_schema_config import get_custom_schema, get_custom_instructions
# Import the prompts
from extraction_prompts import (
    # Material Properties
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,
    # Physical / Mechanical Attributes
    PULL_TO_SEAT_PROMPT,
    GENDER_PROMPT,
    HEIGHT_MM_PROMPT,
    LENGTH_MM_PROMPT,
    WIDTH_MM_PROMPT,
    NUMBER_OF_CAVITIES_PROMPT,
    NUMBER_OF_ROWS_PROMPT,
    MECHANICAL_CODING_PROMPT,
    COLOUR_PROMPT,
    COLOUR_CODING_PROMPT,
    # Sealing & Environmental
    WORKING_TEMPERATURE_PROMPT,
    HOUSING_SEAL_PROMPT,
    WIRE_SEAL_PROMPT,
    SEALING_PROMPT,
    SEALING_CLASS_PROMPT,
    # Terminals & Connections
    CONTACT_SYSTEMS_PROMPT,
    TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT,
    CLOSED_CAVITIES_PROMPT,
    # Assembly & Type
    PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT,
    SET_KIT_PROMPT,
    # Specialized Attributes
    HV_QUALIFIED_PROMPT
)
# Import the web prompts
from extraction_prompts_web import (
    # Material Properties
    MATERIAL_FILLING_WEB_PROMPT,
    MATERIAL_NAME_WEB_PROMPT,
    # Physical / Mechanical Attributes
    PULL_TO_SEAT_WEB_PROMPT,
    GENDER_WEB_PROMPT,
    HEIGHT_MM_WEB_PROMPT,
    LENGTH_MM_WEB_PROMPT,
    WIDTH_MM_WEB_PROMPT,
    NUMBER_OF_CAVITIES_WEB_PROMPT,
    NUMBER_OF_ROWS_WEB_PROMPT,
    MECHANICAL_CODING_WEB_PROMPT,
    COLOUR_WEB_PROMPT,
    COLOUR_CODING_WEB_PROMPT,
    # Sealing & Environmental
    MAX_WORKING_TEMPERATURE_WEB_PROMPT,
    MIN_WORKING_TEMPERATURE_WEB_PROMPT,
    HOUSING_SEAL_WEB_PROMPT,
    WIRE_SEAL_WEB_PROMPT,
    SEALING_WEB_PROMPT,
    SEALING_CLASS_WEB_PROMPT,
    # Terminals & Connections
    CONTACT_SYSTEMS_WEB_PROMPT,
    TERMINAL_POSITION_ASSURANCE_WEB_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT,
    CLOSED_CAVITIES_WEB_PROMPT,
    # Assembly & Type
    PRE_ASSEMBLED_WEB_PROMPT,
    CONNECTOR_TYPE_WEB_PROMPT,
    SET_KIT_WEB_PROMPT,
    # Specialized Attributes
    HV_QUALIFIED_WEB_PROMPT
)

# --- Define the prompts dictionary ---
prompts_to_run = { 
    # Material Properties
    "Material Filling": {"pdf": MATERIAL_PROMPT, "web": MATERIAL_FILLING_WEB_PROMPT},
    "Material Name": {"pdf": MATERIAL_NAME_PROMPT, "web": MATERIAL_NAME_WEB_PROMPT},
    # Physical / Mechanical Attributes
    "Pull-To-Seat": {"pdf": PULL_TO_SEAT_PROMPT, "web": PULL_TO_SEAT_WEB_PROMPT},
    "Gender": {"pdf": GENDER_PROMPT, "web": GENDER_WEB_PROMPT},
    "Height [MM]": {"pdf": HEIGHT_MM_PROMPT, "web": HEIGHT_MM_WEB_PROMPT},
    "Length [MM]": {"pdf": LENGTH_MM_PROMPT, "web": LENGTH_MM_WEB_PROMPT},
    "Width [MM]": {"pdf": WIDTH_MM_PROMPT, "web": WIDTH_MM_WEB_PROMPT},
    "Number Of Cavities": {"pdf": NUMBER_OF_CAVITIES_PROMPT, "web": NUMBER_OF_CAVITIES_WEB_PROMPT},
    "Number Of Rows": {"pdf": NUMBER_OF_ROWS_PROMPT, "web": NUMBER_OF_ROWS_WEB_PROMPT},
    "Mechanical Coding": {"pdf": MECHANICAL_CODING_PROMPT, "web": MECHANICAL_CODING_WEB_PROMPT},
    "Colour": {"pdf": COLOUR_PROMPT, "web": COLOUR_WEB_PROMPT},
    "Colour Coding": {"pdf": COLOUR_CODING_PROMPT, "web": COLOUR_CODING_WEB_PROMPT},
    # Sealing & Environmental
    "Max. Working Temperature [°C]": {"pdf": WORKING_TEMPERATURE_PROMPT, "web": MAX_WORKING_TEMPERATURE_WEB_PROMPT},
    "Min. Working Temperature [°C]": {"pdf": WORKING_TEMPERATURE_PROMPT, "web": MIN_WORKING_TEMPERATURE_WEB_PROMPT},
    "Housing Seal": {"pdf": HOUSING_SEAL_PROMPT, "web": HOUSING_SEAL_WEB_PROMPT},
    "Wire Seal": {"pdf": WIRE_SEAL_PROMPT, "web": WIRE_SEAL_WEB_PROMPT},
    "Sealing": {"pdf": SEALING_PROMPT, "web": SEALING_WEB_PROMPT},
    "Sealing Class": {"pdf": SEALING_CLASS_PROMPT, "web": SEALING_CLASS_WEB_PROMPT},
    # Terminals & Connections
    "Contact Systems": {"pdf": CONTACT_SYSTEMS_PROMPT, "web": CONTACT_SYSTEMS_WEB_PROMPT},
    "Terminal Position Assurance": {"pdf": TERMINAL_POSITION_ASSURANCE_PROMPT, "web": TERMINAL_POSITION_ASSURANCE_WEB_PROMPT},
    "Connector Position Assurance": {"pdf": CONNECTOR_POSITION_ASSURANCE_PROMPT, "web": CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT},
    "Name Of Closed Cavities": {"pdf": CLOSED_CAVITIES_PROMPT, "web": CLOSED_CAVITIES_WEB_PROMPT},
    # Assembly & Type
    "Pre-assembled": {"pdf": PRE_ASSEMBLED_PROMPT, "web": PRE_ASSEMBLED_WEB_PROMPT},
    "Type Of Connector": {"pdf": CONNECTOR_TYPE_PROMPT, "web": CONNECTOR_TYPE_WEB_PROMPT},
    "Set/Kit": {"pdf": SET_KIT_PROMPT, "web": SET_KIT_WEB_PROMPT},
    # Specialized Attributes
    "HV Qualified": {"pdf": HV_QUALIFIED_PROMPT, "web": HV_QUALIFIED_WEB_PROMPT}
}

# --- Application State ---
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'pdf_chain' not in st.session_state:
    st.session_state.pdf_chain = None
if 'web_chain' not in st.session_state:
    st.session_state.web_chain = None
if 'numind_chain' not in st.session_state:
    st.session_state.numind_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None
if 'extraction_performed' not in st.session_state:
    st.session_state.extraction_performed = False
if 'extraction_attempts' not in st.session_state:
    st.session_state.extraction_attempts = 0
if 'scraped_table_html_cache' not in st.session_state:
    st.session_state.scraped_table_html_cache = None
if 'current_part_number_scraped' not in st.session_state:
    st.session_state.current_part_number_scraped = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'uploaded_file_data' not in st.session_state:
    st.session_state.uploaded_file_data = []

def reset_evaluation_state():
    """Reset all evaluation-related session state variables."""
    st.session_state.evaluation_results = []
    st.session_state.evaluation_metrics = None
    st.session_state.extraction_performed = False
    st.session_state.scraped_table_html_cache = None
    st.session_state.current_part_number_scraped = None
    st.session_state.processed_documents = []
    st.session_state.uploaded_file_data = []

# --- Global Variables / Initialization ---
@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings function."""
    embeddings = get_embedding_function()
    return embeddings

@st.cache_resource
def initialize_llm_cached():
    """Initialize LLM function."""
    llm_instance = initialize_llm()
    return llm_instance

# --- Initialize Core Components ---
embedding_function = None
llm = None

try:
    logger.info("Attempting to initialize embedding function...")
    embedding_function = initialize_embeddings()
    if embedding_function:
         logger.success("Embedding function initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
    st.error(f"Fatal Error: Could not initialize embedding model. Error: {e}")
    st.stop()

try:
    logger.info("Attempting to initialize LLM...")
    llm = initialize_llm_cached()
    if llm:
        logger.success("LLM initialized successfully.")
except Exception as e:
     logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
     st.error(f"Fatal Error: Could not initialize LLM. Error: {e}")
     st.stop()

if embedding_function is None or llm is None:
    st.error("Core components (Embeddings or LLM) failed to initialize. Cannot continue.")
    st.stop()

# --- UI Layout ---
# Blue band header with LEONI


# Blue band header with LEONI
st.markdown("""
    <div class="header-band">
        <h1>LEONI</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📄 PDF Attribute Extraction")
st.markdown("Upload your PDF documents and automatically extract key attributes.")

if not config.GROQ_API_KEY:
    st.warning("Groq API Key not found. Please set the GROQ_API_KEY environment variable.", icon="⚠️")

# --- Sidebar for PDF Upload and Processing ---
with st.sidebar:
    st.header("1. Document Processing")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    # --- Add Part Number Input ---
    st.text_input("Enter Part Number (Optional):", key="part_number_input", value=st.session_state.get("part_number_input", ""))
    # ---------------------------

    process_button = st.button("Process Uploaded Documents", key="process_button", type="primary")

    if process_button and uploaded_files:
        if not embedding_function or not llm:
             st.error("Core components (Embeddings or LLM) failed to initialize earlier. Cannot process documents.")
        else:
            # Reset state including evaluation and the extraction flag
            st.session_state.retriever = None
            # Reset BOTH chains
            st.session_state.pdf_chain = None
            st.session_state.web_chain = None
            st.session_state.processed_files = []
            reset_evaluation_state() # Reset evaluation results AND extraction flag

            filenames = [f.name for f in uploaded_files]
            # Store uploaded file data for NuMind extraction
            st.session_state.uploaded_file_data = [(f.name, f.getvalue()) for f in uploaded_files]
            logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)}")
            # --- PDF Processing ---
            with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
                processed_docs = [] # Initialize as empty list instead of None
                try:
                    start_time = time.time()
                    temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                    
                    # Create event loop for async processing
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Call the async function properly
                    processed_docs = loop.run_until_complete(process_uploaded_pdfs(uploaded_files, temp_dir))
                    
                    processing_time = time.time() - start_time
                    logger.info(f"PDF processing took {processing_time:.2f} seconds.")
                except Exception as e:
                    logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                    st.error(f"Error processing PDFs: {e}")
                    processed_docs = [] # Ensure it's an empty list on error

            # --- Vector Store Indexing ---
            if processed_docs and len(processed_docs) > 0:
                logger.info(f"Generated {len(processed_docs)} documents.")
                with st.spinner("Indexing documents in vector store..."):
                    try:
                        start_time = time.time()
                        st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                        indexing_time = time.time() - start_time
                        logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")

                        if st.session_state.retriever:
                            st.session_state.processed_files = filenames # Update list
                            st.session_state.processed_documents = processed_docs # Store the Mistral-extracted documents
                            logger.success("Vector store setup complete. Retriever is ready.")
                            # --- Create Extraction Chains --- 
                            with st.spinner("Preparing extraction engines..."):
                                 st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                                 st.session_state.web_chain = create_web_extraction_chain(llm)
                                 st.session_state.numind_chain = create_numind_extraction_chain()
                            if st.session_state.pdf_chain and st.session_state.web_chain:
                                logger.success("Extraction chains created.")
                                # Keep extraction_performed as False here, it will run in the main section
                                st.success(f"Successfully processed {len(filenames)} file(s). Evaluation below.") # Update message
                            else:
                                st.error("Failed to create one or both extraction chains after processing.")
                                # reset_evaluation_state() called earlier is sufficient
                        else:
                            st.error("Failed to setup vector store after processing PDFs.")
                            # reset_evaluation_state() called earlier is sufficient
                    except Exception as e:
                         logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                         st.error(f"Error setting up vector store: {e}")
                         # reset_evaluation_state() called earlier is sufficient
            elif not processed_docs and uploaded_files:
                st.warning("No text could be extracted or processed from the uploaded PDFs.")
                # reset_evaluation_state() called earlier is sufficient

    elif process_button and not uploaded_files:
        st.warning("Please upload at least one PDF file before processing.")

    # --- Display processed files status (Simplified) ---
    st.subheader("Processing Status")
    # Check if both chains are ready for the full process
    if st.session_state.pdf_chain and st.session_state.web_chain and st.session_state.processed_files:
        st.success(f"Ready. Processed: {', '.join(st.session_state.processed_files)}")
    else:
        st.info("Upload and process PDF documents to view extracted data.")


# --- Main Layout with Two Columns ---
# Initialize chatbot


# Create two columns: left for extraction, right for results and chat
# left_col, right_col = st.columns([2, 1])
# with left_col:
#     st.header("2. Extracted Information")

st.header("2. Extracted Information")

# --- Get current asyncio event loop --- 
# Needed for both scraping and running the async extraction chain
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # 'RuntimeError: There is no current event loop...'
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# -------------------------------------

# Check if BOTH chains are ready before proceeding
if not st.session_state.pdf_chain or not st.session_state.web_chain:
    st.info("Upload and process documents using the sidebar to see extracted results here.")
    # Ensure evaluation state is also clear if no chain
    if not st.session_state.evaluation_results and not st.session_state.extraction_performed:
         reset_evaluation_state() # Ensure reset if no chain and extraction not done
else:
    # Add debug logging
    logger.debug(f"PDF Chain: {st.session_state.pdf_chain is not None}")
    logger.debug(f"Web Chain: {st.session_state.web_chain is not None}")
    logger.debug(f"Extraction Performed: {st.session_state.extraction_performed}")
    logger.debug(f"Evaluation Results: {len(st.session_state.evaluation_results) if st.session_state.evaluation_results else 0}")
    
    # --- Block 1: Run Extraction (if needed) --- 
    if (st.session_state.pdf_chain and st.session_state.web_chain) and not st.session_state.extraction_performed:
        # Safety check to prevent infinite loops
        if st.session_state.extraction_attempts > 3:
            logger.error("Too many extraction attempts detected. Resetting state.")
            st.error("Extraction loop detected. Please refresh the page and try again.")
            reset_evaluation_state()
            st.session_state.extraction_attempts = 0
            st.stop()
        
        st.session_state.extraction_attempts += 1
        logger.info(f"Starting extraction process... (attempt {st.session_state.extraction_attempts})")
        
        # --- Get Part Number --- 
        part_number = st.session_state.get("part_number_input", "").strip()
        # ---------------------
        
        # --- DEBUG: Log extraction start ---
        debug_logger.user_action("Extract button clicked", data={
            "attempt": st.session_state.extraction_attempts,
            "part_number": part_number,
            "session_state_keys": list(st.session_state.keys())
        }, context={"page": "extraction_attributs"})
        
        debug_logger.info("Part number retrieved", data={"part_number": part_number}, context={"step": "part_number_retrieval"})

        # --- Block 1a: Scrape Web Table HTML (if needed) --- 
        scraped_table_html = None # Initialize
        if part_number: # Only scrape if part number is provided
            debug_logger.info("Starting web scraping", data={"part_number": part_number}, context={"step": "web_scraping_start"})
            
            # Check cache first
            if st.session_state.current_part_number_scraped == part_number and st.session_state.scraped_table_html_cache is not None:
                 logger.info(f"Using cached scraped HTML for part number {part_number}.")
                 scraped_table_html = st.session_state.scraped_table_html_cache
                 debug_logger.info("Using cached web data", data={
                     "part_number": part_number,
                     "cached_html_length": len(scraped_table_html) if scraped_table_html else 0
                 }, context={"step": "web_scraping_cache_hit"})
            else:
                 # Scrape and update cache
                 logger.info(f"Part number {part_number} changed or not cached. Attempting web scrape...")
                 debug_logger.info("Cache miss, starting web scrape", data={
                     "part_number": part_number,
                     "cached_part": st.session_state.current_part_number_scraped
                 }, context={"step": "web_scraping_cache_miss"})
                 
                 with st.spinner("Attempting to scrape data from supplier websites..."):
                     scrape_start_time = time.time()
                     try:
                          # Ensure scrape_website_table_html is imported from llm_interface
                          from llm_interface import scrape_website_table_html
                          
                          debug_logger.llm_request(
                              f"Scraping web data for part {part_number}",
                              "web_scraper",
                              0.0,
                              0,
                              context={"step": "web_scraping_request"}
                          )
                          
                          scraped_table_html = loop.run_until_complete(scrape_website_table_html(part_number))
                          scrape_time = time.time() - scrape_start_time
                          
                          debug_logger.web_scraping(
                              f"Part {part_number}",
                              scraped_table_html if scraped_table_html else "",
                              scraped_table_html,
                              context={"step": "web_scraping_response", "duration": scrape_time}
                          )
                          
                          if scraped_table_html:
                              logger.success(f"Web scraping successful in {scrape_time:.2f} seconds.")
                              st.caption(f"ℹ️ Found web data for part# {part_number}. Will prioritize.")
                              debug_logger.info("Web scraping successful", data={
                                  "duration": scrape_time,
                                  "html_length": len(scraped_table_html)
                              }, context={"step": "web_scraping_success"})
                          else:
                              logger.warning(f"Web scraping attempted but failed to find table HTML in {scrape_time:.2f} seconds.")
                              st.caption(f"⚠️ Web scraping failed for part# {part_number}, using PDF data only.")
                              debug_logger.warning("Web scraping failed", data={
                                  "duration": scrape_time,
                                  "reason": "No table HTML found"
                              }, context={"step": "web_scraping_failed"})
                          
                          # Update cache
                          st.session_state.scraped_table_html_cache = scraped_table_html
                          st.session_state.current_part_number_scraped = part_number
                          debug_logger.session_state("scraped_table_html_cache", scraped_table_html, context={"step": "web_scraping_cache_update"})
                          debug_logger.session_state("current_part_number_scraped", part_number, context={"step": "web_scraping_cache_update"})
                          
                     except Exception as scrape_e:
                          scrape_time = time.time() - scrape_start_time
                          logger.error(f"Error during web scraping ({scrape_time:.2f}s): {scrape_e}", exc_info=True)
                          st.warning(f"An error occurred during web scraping: {scrape_e}. Using PDF data only.")
                          
                          debug_logger.exception(scrape_e, context={
                              "step": "web_scraping_exception",
                              "duration": scrape_time,
                              "part_number": part_number
                          })
                          
                          # Ensure cache is cleared on error
                          st.session_state.scraped_table_html_cache = None
                          st.session_state.current_part_number_scraped = part_number
                          debug_logger.session_state("scraped_table_html_cache", None, context={"step": "web_scraping_cache_clear"})
        else:
             logger.info("No part number provided, skipping web scrape.")
             debug_logger.info("Skipping web scrape", data={"reason": "No part number provided"}, context={"step": "web_scraping_skipped"})
             # Clear cache if part number is removed
             if st.session_state.current_part_number_scraped is not None:
                  st.session_state.scraped_table_html_cache = None
                  st.session_state.current_part_number_scraped = None
                  debug_logger.session_state("scraped_table_html_cache", None, context={"step": "web_scraping_cache_clear_no_part"})
                  debug_logger.session_state("current_part_number_scraped", None, context={"step": "web_scraping_cache_clear_no_part"})
        # --- End Block 1a ---

        # --- Log the result of scraping before Stage 1 --- 
        logger.debug(f"Cleaned Scraped HTML content passed to Stage 1: {scraped_table_html[:500] if scraped_table_html else 'None'}...")
        debug_logger.info("Web scraping completed", data={
            "has_html": scraped_table_html is not None,
            "html_length": len(scraped_table_html) if scraped_table_html else 0,
            "html_preview": scraped_table_html[:1000] if scraped_table_html else None
        }, context={"step": "web_scraping_complete"})
        # -------------------------------------------------

        # --- Block 1b: Three-Stage Extraction Logic --- 
        st.info(f"Running Stage 1 (Web Data Extraction) for {len(prompts_to_run)} attributes...")
        
        # Progress indicator for three-stage process
        progress_col1, progress_col2, progress_col3 = st.columns(3)
        with progress_col1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                            color: white; 
                            padding: 0.5rem; 
                            border-radius: 10px; 
                            text-align: center; 
                            margin-bottom: 1rem;">
                    <strong>Stage 1: Web</strong><br>
                    <small>Web scraping & extraction</small>
                </div>
            """, unsafe_allow_html=True)
        with progress_col2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                            color: white; 
                            padding: 0.5rem; 
                            border-radius: 10px; 
                            text-align: center; 
                            margin-bottom: 1rem;">
                    <strong>Stage 2: NuMind</strong><br>
                    <small>Structured extraction</small>
                </div>
            """, unsafe_allow_html=True)
        with progress_col3:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%); 
                            color: white; 
                            padding: 0.5rem; 
                            border-radius: 10px; 
                            text-align: center; 
                            margin-bottom: 1rem;">
                    <strong>Stage 3: Fallback</strong><br>
                    <small>Final recheck</small>
                </div>
            """, unsafe_allow_html=True)
        
        cols = st.columns(2) # For displaying progress
        col_index = 0
        SLEEP_INTERVAL_SECONDS = 0.2 # Can potentially be lower for web chain
        
        intermediate_results = {} # Store stage 1 results {prompt_name: {result_data}} 
        pdf_fallback_needed = [] # List of prompt_names needing stage 2

        # --- Stage 1: Web Extraction --- 
        if scraped_table_html:
            debug_logger.info("Starting Stage 1 (Web Extraction)", data={
                "total_attributes": len(prompts_to_run),
                "has_html": True,
                "html_length": len(scraped_table_html)
            }, context={"step": "stage1_start"})
            
            for prompt_name, instructions in prompts_to_run.items(): # Iterate through attributes and their instructions
                attribute_key = prompt_name
                web_instruction = instructions["web"] # Get WEB instruction
                current_col = cols[col_index % 2]
                col_index += 1
                json_result_str = None
                run_time = 0.0
                source = "Web" # Source for this stage
                
                debug_logger.info(f"Processing attribute: {attribute_key}", data={
                    "attribute": attribute_key,
                    "source": source,
                    "instruction_length": len(web_instruction)
                }, context={"step": "stage1_attribute_start", "attribute": attribute_key})
                
                with current_col:
                     with st.spinner(f"Stage 1: Extracting {attribute_key} from Web Data..."):
                        try:
                            start_time = time.time()
                            web_input = {
                                "cleaned_web_data": scraped_table_html,
                                "attribute_key": attribute_key,
                                "extraction_instructions": web_instruction # Use specific web instruction
                            }
                            
                            debug_logger.llm_request(
                                f"Extract {attribute_key} from web data",
                                "web_chain",
                                0.7,
                                1000,
                                context={"step": "stage1_llm_request", "attribute": attribute_key}
                            )
                            
                            # --- Log the input to the web chain --- 
                            logger.debug(f"Invoking web_chain for '{attribute_key}' with input keys: {list(web_input.keys())}")
                            # -------------------------------------
                            # Call helper using the web_chain
                            json_result_str = loop.run_until_complete(
                                _invoke_chain_and_process(st.session_state.web_chain, web_input, f"{attribute_key} (Web)")
                            )
                            run_time = time.time() - start_time
                            
                            debug_logger.llm_response(
                                "web_chain",
                                json_result_str if json_result_str else "",
                                len(json_result_str) if json_result_str else 0,
                                run_time,
                                context={"step": "stage1_llm_response", "attribute": attribute_key}
                            )
                            
                            logger.info(f"Stage 1 (Web) for '{attribute_key}' took {run_time:.2f} seconds.")
                            time.sleep(SLEEP_INTERVAL_SECONDS) # Add delay
                        except Exception as e:
                             logger.error(f"Error during Stage 1 (Web) call for '{attribute_key}': {e}", exc_info=True)
                             json_result_str = f'{{"error": "Exception during Stage 1 call: {e}"}}'
                             run_time = time.time() - start_time # Record time even on error
                             
                             debug_logger.exception(e, context={
                                 "step": "stage1_exception",
                                 "attribute": attribute_key,
                                 "duration": run_time
                             })
                
                # --- Log the raw output from the web chain ---
                logger.debug(f"Raw JSON result string from web_chain for '{attribute_key}': {json_result_str}")
                debug_logger.info(f"Raw output for {attribute_key}", data={
                    "raw_output": json_result_str,
                    "output_length": len(json_result_str) if json_result_str else 0
                }, context={"step": "stage1_raw_output", "attribute": attribute_key})
                # -----------------------------------------
                
                # --- Basic Parsing of Stage 1 Result --- 
                final_answer_value = "Error"
                parse_error = None
                is_rate_limit = False
                llm_returned_error_msg = None
                raw_output = json_result_str if json_result_str else '{"error": "Stage 1 did not run"}'
                needs_fallback = False
                
                try:
                    string_to_parse = raw_output.strip()
                    parsed_json = extract_json_from_string(string_to_parse)
                    
                    debug_logger.data_transformation(
                        f"JSON parsing for {attribute_key}",
                        string_to_parse,
                        parsed_json,
                        context={"step": "stage1_json_parsing", "attribute": attribute_key}
                    )
                    
                    if not isinstance(parsed_json, dict):
                        logger.error(f"Stage 1: Parsed JSON is not a dict for '{attribute_key}'. Got: {parsed_json}. Raw: {string_to_parse}")
                        final_answer_value = "Unexpected JSON Type"
                        parse_error = TypeError(f"Stage 1 Expected dict, got {type(parsed_json)}")
                        needs_fallback = True
                        debug_logger.warning(f"Unexpected JSON type for {attribute_key}", data={
                            "expected": "dict",
                            "got": type(parsed_json).__name__,
                            "parsed_value": parsed_json
                        }, context={"step": "stage1_parse_error", "attribute": attribute_key})
                    elif attribute_key in parsed_json:
                        parsed_value = str(parsed_json[attribute_key])
                        # Check for NOT FOUND variants 
                        if "not found" in parsed_value.lower() or parsed_value.strip() == "":
                            final_answer_value = "NOT FOUND"
                            needs_fallback = True # Mark for PDF stage
                            logger.info(f"Stage 1 result for '{attribute_key}' is NOT FOUND. Queued for PDF fallback.")
                            debug_logger.info(f"NOT FOUND for {attribute_key}", data={
                                "parsed_value": parsed_value,
                                "needs_fallback": True
                            }, context={"step": "stage1_not_found", "attribute": attribute_key})
                        else:
                            final_answer_value = parsed_value # Store successful web result
                            logger.success(f"Stage 1 successful for '{attribute_key}' from Web data.")
                            debug_logger.extraction_step(
                                attribute_key,
                                source,
                                web_input,
                                final_answer_value,
                                True,
                                context={"step": "stage1_success", "attribute": attribute_key}
                            )
                    elif "error" in parsed_json:
                        error_msg = parsed_json['error']
                        llm_returned_error_msg = error_msg
                        if "rate limit" in error_msg.lower():
                            final_answer_value = "Rate Limit Hit"
                            is_rate_limit = True
                            parse_error = ValueError("Rate limit hit (Web)")
                            debug_logger.warning(f"Rate limit hit for {attribute_key}", data={
                                "error_msg": error_msg
                            }, context={"step": "stage1_rate_limit", "attribute": attribute_key})
                        else:
                            final_answer_value = f"Error: {error_msg[:100]}"
                            parse_error = ValueError(f"Stage 1 Error: {error_msg}")
                            debug_logger.warning(f"LLM error for {attribute_key}", data={
                                "error_msg": error_msg
                            }, context={"step": "stage1_llm_error", "attribute": attribute_key})
                        needs_fallback = True # Also fallback on web chain error
                        logger.warning(f"Stage 1 Error for '{attribute_key}'. Queued for PDF fallback. Error: {error_msg}")
                    else:
                        final_answer_value = "Unexpected JSON Format"
                        parse_error = ValueError(f"Stage 1 Unexpected JSON keys: {list(parsed_json.keys())}")
                        needs_fallback = True # Fallback on unexpected format
                        debug_logger.warning(f"Unexpected JSON format for {attribute_key}", data={
                            "found_keys": list(parsed_json.keys()),
                            "expected_key": attribute_key
                        }, context={"step": "stage1_unexpected_format", "attribute": attribute_key})
                except Exception as processing_exc:
                    parse_error = processing_exc
                    final_answer_value = "Processing Error"
                    logger.error(f"Error processing Stage 1 result for '{attribute_key}'. Error: {processing_exc}. Raw: {string_to_parse}")
                    debug_logger.exception(processing_exc, context={
                        "step": "stage1_processing_exception",
                        "attribute": attribute_key,
                        "raw_output": string_to_parse
                    })
                
                # Store intermediate result (even if NOT FOUND or error)
                is_error = bool(parse_error) and not is_rate_limit
                is_not_found_stage1 = final_answer_value == "NOT FOUND"
                is_success_stage1 = not is_error and not is_not_found_stage1 and not is_rate_limit
                
                intermediate_results[prompt_name] = {
                    'Prompt Name': prompt_name,
                    'Extracted Value': final_answer_value, # Store Stage 1 value/error/NOT FOUND
                    'Source': source,
                    'Raw Output': raw_output,
                    'Parse Error': str(parse_error) if parse_error else None,
                    'Is Success': is_success_stage1,
                    'Is Error': is_error,
                    'Is Not Found': is_not_found_stage1,
                    'Is Rate Limit': is_rate_limit,
                    'Latency (s)': round(run_time, 2),
                    'Exact Match': None,
                    'Case-Insensitive Match': None
                }
                
                debug_logger.info(f"Stage 1 result stored for {attribute_key}", data={
                    "final_value": final_answer_value,
                    "success": is_success_stage1,
                    "error": is_error,
                    "not_found": is_not_found_stage1,
                    "rate_limit": is_rate_limit,
                    "latency": round(run_time, 2)
                }, context={"step": "stage1_result_stored", "attribute": attribute_key})
                
                if needs_fallback:
                    pdf_fallback_needed.append(prompt_name)
                    debug_logger.info(f"Added {attribute_key} to PDF fallback list", context={"step": "stage1_fallback_queued", "attribute": attribute_key})
        
        else: # No scraped HTML, all attributes need PDF fallback
            logger.info("No scraped web data available. All attributes will use PDF extraction.")
            debug_logger.info("No web data, all attributes need PDF fallback", data={
                "total_attributes": len(prompts_to_run),
                "fallback_list": list(prompts_to_run.keys())
            }, context={"step": "stage1_skipped_no_web_data"})
            
            pdf_fallback_needed = list(prompts_to_run.keys())
            # Populate intermediate results with placeholders indicating skipped web stage
            for prompt_name in pdf_fallback_needed:
                 intermediate_results[prompt_name] = {
                    'Prompt Name': prompt_name,
                    'Extracted Value': "(Web Stage Skipped)", 
                    'Source': 'Pending',
                    'Raw Output': 'N/A',
                    'Parse Error': None,
                    'Is Success': False,
                    'Is Error': False,
                    'Is Not Found': True,
                    'Is Rate Limit': False,
                    'Latency (s)': 0.0,
                    'Exact Match': None,
                    'Case-Insensitive Match': None
                }

        # --- Stage 2: NuMind Fallback --- 
        st.info(f"Running Stage 2 (NuMind Fallback) for {len(pdf_fallback_needed)} attributes...")
        debug_logger.info("Starting Stage 2 (NuMind Fallback)", data={
            "fallback_count": len(pdf_fallback_needed),
            "fallback_attributes": pdf_fallback_needed
        }, context={"step": "stage2_start"})
        
        col_index = 0 # Reset column index
        SLEEP_INTERVAL_SECONDS = 0.5 # Potentially longer delay for NuMind API calls

        if not pdf_fallback_needed:
            st.success("Stage 1 extraction successful for all attributes from web data.")
            debug_logger.info("No NuMind fallback needed", data={
                "reason": "All attributes successful in Stage 1"
            }, context={"step": "stage2_skipped_all_successful"})
        else:
            # Check if NuMind is available
            if not st.session_state.numind_chain:
                st.warning("NuMind extraction not available. Falling back to PDF extraction.")
                debug_logger.warning("NuMind not available, using PDF fallback", context={"step": "stage2_numind_unavailable"})
                
                # Fallback to original PDF extraction logic
                for prompt_name in pdf_fallback_needed:
                    attribute_key = prompt_name
                    pdf_instruction = prompts_to_run[attribute_key]["pdf"]
                    current_col = cols[col_index % 2]
                    col_index += 1
                    json_result_str = None
                    run_time = 0.0
                    source = "PDF"
                    
                    with current_col:
                        with st.spinner(f"Stage 2: Extracting {attribute_key} from PDF Data..."):
                            try:
                                start_time = time.time()
                                context_chunks = fetch_chunks(
                                    st.session_state.retriever,
                                    part_number,
                                    attribute_key,
                                    k=8
                                )
                                context_text = "\n\n".join([chunk.page_content for chunk in context_chunks]) if context_chunks else ""
                                
                                pdf_input = {
                                    "context": context_text,
                                    "extraction_instructions": pdf_instruction,
                                    "attribute_key": attribute_key,
                                    "part_number": part_number if part_number else "Not Provided"
                                }
                                
                                json_result_str = loop.run_until_complete(
                                    _invoke_chain_and_process(st.session_state.pdf_chain, pdf_input, f"{attribute_key} (PDF)")
                                )
                                run_time = time.time() - start_time
                                time.sleep(SLEEP_INTERVAL_SECONDS)
                            except Exception as e:
                                logger.error(f"Error during Stage 2 (PDF) call for '{attribute_key}': {e}", exc_info=True)
                                json_result_str = f'{{"error": "Exception during Stage 2 call: {e}"}}'
                                run_time = time.time() - start_time
                    
                    # Parse PDF result (same logic as before)
                    final_answer_value = "Error"
                    parse_error = None
                    is_rate_limit = False
                    raw_output = json_result_str if json_result_str else '{"error": "Stage 2 did not run"}'
                    
                    try:
                        string_to_parse = raw_output.strip()
                        parsed_json = extract_json_from_string(string_to_parse)
                        
                        if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                            final_answer_value = str(parsed_json[attribute_key])
                        elif isinstance(parsed_json, dict) and "error" in parsed_json:
                            final_answer_value = f"Error: {parsed_json['error'][:100]}"
                            parse_error = ValueError(f"Stage 2 Error: {parsed_json['error']}")
                        else:
                            final_answer_value = "Unexpected JSON Format"
                            parse_error = ValueError(f"Stage 2 Unexpected JSON format")
                    except Exception as processing_exc:
                        parse_error = processing_exc
                        final_answer_value = "Processing Error"
                    
                    # Update results
                    is_error = bool(parse_error) and not is_rate_limit
                    is_not_found_stage2 = "not found" in final_answer_value.lower() or final_answer_value.strip() == ""
                    is_success_stage2 = not is_error and not is_not_found_stage2 and not is_rate_limit
                    
                    stage1_latency = intermediate_results[prompt_name].get('Latency (s)', 0.0)
                    total_latency = stage1_latency + round(run_time, 2)
                    
                    intermediate_results[prompt_name].update({
                        'Extracted Value': final_answer_value,
                        'Source': source,
                        'Raw Output': raw_output,
                        'Parse Error': str(parse_error) if parse_error else None,
                        'Is Success': is_success_stage2,
                        'Is Error': is_error,
                        'Is Not Found': is_not_found_stage2,
                        'Is Rate Limit': is_rate_limit,
                        'Latency (s)': total_latency 
                    })
            else:
                # Use NuMind extraction
                st.success("Using NuMind for structured extraction...")
                
                # Get the first uploaded file data for NuMind extraction
                file_data = None
                if st.session_state.uploaded_file_data and len(st.session_state.uploaded_file_data) > 0:
                    # Use the first file for extraction
                    file_name, file_bytes = st.session_state.uploaded_file_data[0]
                    file_data = file_bytes
                
                if not file_data:
                    st.error("No file data available for NuMind extraction.")
                    debug_logger.error("No file data for NuMind", context={"step": "stage2_no_file_data"})
                else:
                    # Run NuMind extraction once for all attributes
                    with st.spinner("Running NuMind structured extraction..."):
                        try:
                            start_time = time.time()
                            
                            debug_logger.info("Starting NuMind extraction", data={
                                "file_name": st.session_state.uploaded_file_data[0][0] if st.session_state.uploaded_file_data else "Unknown",
                                "file_size": len(file_data),
                                "attributes_count": len(pdf_fallback_needed)
                            }, context={"step": "stage2_numind_start"})
                            
                            # Get the custom extraction schema that matches your NuMind playground
                            extraction_schema = get_custom_schema()
                            
                            # Run NuMind extraction with your custom schema
                            numind_result = loop.run_until_complete(
                                extract_with_numind_using_schema(st.session_state.numind_chain, file_data, extraction_schema)
                            )
                            
                            run_time = time.time() - start_time
                            
                            debug_logger.info("NuMind extraction completed", data={
                                "duration": run_time,
                                "result_keys": list(numind_result.keys()) if numind_result else []
                            }, context={"step": "stage2_numind_complete"})
                            
                            if numind_result:
                                st.success(f"NuMind extraction completed in {run_time:.2f} seconds.")
                                
                                # Process each attribute from NuMind result
                                for prompt_name in pdf_fallback_needed:
                                    attribute_key = prompt_name
                                    current_col = cols[col_index % 2]
                                    col_index += 1
                                    source = "NuMind"
                                    
                                    debug_logger.info(f"Processing NuMind result for: {attribute_key}", context={"step": "stage2_numind_attribute", "attribute": attribute_key})
                                    
                                    # Extract specific attribute from NuMind result
                                    final_answer_value = extract_specific_attribute_from_numind_result(numind_result, attribute_key)
                                    
                                    if final_answer_value is None:
                                        final_answer_value = "NOT FOUND"
                                        is_success_stage2 = False
                                        is_error = False
                                        is_not_found_stage2 = True
                                        parse_error = None
                                    else:
                                        is_success_stage2 = True
                                        is_error = False
                                        is_not_found_stage2 = False
                                        parse_error = None
                                    
                                    # Update results
                                    stage1_latency = intermediate_results[prompt_name].get('Latency (s)', 0.0)
                                    total_latency = stage1_latency + round(run_time, 2)
                                    
                                    intermediate_results[prompt_name].update({
                                        'Extracted Value': final_answer_value,
                                        'Source': source,
                                        'Raw Output': json.dumps(numind_result) if numind_result else "No NuMind result",
                                        'Parse Error': None,
                                        'Is Success': is_success_stage2,
                                        'Is Error': is_error,
                                        'Is Not Found': is_not_found_stage2,
                                        'Is Rate Limit': False,
                                        'Latency (s)': total_latency 
                                    })
                                    
                                    debug_logger.info(f"NuMind result stored for {attribute_key}", data={
                                        "final_value": final_answer_value,
                                        "success": is_success_stage2,
                                        "error": is_error,
                                        "not_found": is_not_found_stage2,
                                        "total_latency": total_latency
                                    }, context={"step": "stage2_numind_result_stored", "attribute": attribute_key})
                                    
                                    logger.info(f"Updated result for '{prompt_name}' with NuMind data.")
                            else:
                                st.error("NuMind extraction failed. No results returned.")
                                debug_logger.error("NuMind extraction failed", context={"step": "stage2_numind_failed"})
                                
                                # Mark all attributes as failed
                                for prompt_name in pdf_fallback_needed:
                                    intermediate_results[prompt_name].update({
                                        'Extracted Value': "NuMind Extraction Failed",
                                        'Source': "NuMind",
                                        'Raw Output': "NuMind API returned no results",
                                        'Parse Error': "NuMind extraction failed",
                                        'Is Success': False,
                                        'Is Error': True,
                                        'Is Not Found': False,
                                        'Is Rate Limit': False,
                                        'Latency (s)': round(run_time, 2)
                                    })
                                
                        except Exception as e:
                            logger.error(f"Error during NuMind extraction: {e}", exc_info=True)
                            st.error(f"NuMind extraction failed: {e}")
                            
                            debug_logger.exception(e, context={
                                "step": "stage2_numind_exception",
                                "duration": time.time() - start_time
                            })
                            
                            # Mark all attributes as failed
                            for prompt_name in pdf_fallback_needed:
                                intermediate_results[prompt_name].update({
                                    'Extracted Value': f"NuMind Error: {str(e)[:100]}",
                                    'Source': "NuMind",
                                    'Raw Output': f"Exception: {e}",
                                    'Parse Error': str(e),
                                    'Is Success': False,
                                    'Is Error': True,
                                    'Is Not Found': False,
                                    'Is Rate Limit': False,
                                    'Latency (s)': round(time.time() - start_time, 2)
                                })

        # --- Final Processing --- 
        # Convert intermediate_results dict to list
        extraction_results_list = list(intermediate_results.values()) 
        
        # --- Stage 3: Final Fallback for NOT FOUND and None Values ---
        # Identify attributes that need final fallback
        final_fallback_needed = []
        for result in extraction_results_list:
            if isinstance(result, dict):
                extracted_value = result.get('Extracted Value', '')
                is_not_found = result.get('Is Not Found', False)
                is_error = result.get('Is Error', False)
                
                # Check for attributes that need final fallback
                if (is_not_found or 
                    extracted_value in ["NOT FOUND", "Error", "Processing Error", "Unexpected JSON Format", "Unexpected JSON Type"] or
                    not extracted_value or 
                    extracted_value.strip() == "" or
                    extracted_value == "(Web Stage Skipped)" or
                    extracted_value.lower() in ["none", "null", "n/a", "na"]):  # Also recheck "none" responses
                    final_fallback_needed.append(result.get('Prompt Name', ''))
        
        if final_fallback_needed:
            # Count how many are "none" responses
            none_responses = []
            other_fallbacks = []
            for result in extraction_results_list:
                if isinstance(result, dict) and result.get('Prompt Name') in final_fallback_needed:
                    extracted_value = result.get('Extracted Value', '')
                    if extracted_value.lower() in ["none", "null", "n/a", "na"]:
                        none_responses.append(result.get('Prompt Name'))
                    else:
                        other_fallbacks.append(result.get('Prompt Name'))
            
            st.info(f"Running Stage 3 (Final Fallback) for {len(final_fallback_needed)} attributes that need rechecking...")
            if none_responses:
                st.warning(f"⚠️ Including {len(none_responses)} attributes that returned 'none' responses - these will be rechecked for potential missed values.")
            
            debug_logger.info("Starting Stage 3 (Final Fallback)", data={
                "fallback_count": len(final_fallback_needed),
                "fallback_attributes": final_fallback_needed,
                "none_responses": none_responses,
                "other_fallbacks": other_fallbacks
            }, context={"step": "stage3_start"})
            
            col_index = 0
            SLEEP_INTERVAL_SECONDS = 0.3
            
            for prompt_name in final_fallback_needed:
                attribute_key = prompt_name
                pdf_instruction = prompts_to_run[attribute_key]["pdf"]
                current_col = cols[col_index % 2]
                col_index += 1
                json_result_str = None
                run_time = 0.0
                source = "Final Fallback"
                
                debug_logger.info(f"Final fallback for attribute: {attribute_key}", context={"step": "stage3_attribute", "attribute": attribute_key})
                
                with current_col:
                    with st.spinner(f"Stage 3: Final recheck for {attribute_key}..."):
                        try:
                            start_time = time.time()
                            
                            # Use more chunks for final fallback to be more thorough
                            context_chunks = fetch_chunks(
                                st.session_state.retriever,
                                part_number,
                                attribute_key,
                                k=12  # Increased from 8 to 12 for more thorough search
                            )
                            context_text = "\n\n".join([chunk.page_content for chunk in context_chunks]) if context_chunks else ""
                            
                            # Enhanced prompt for final fallback
                            # Check if this attribute previously returned "none" or similar
                            previous_value = None
                            for result in extraction_results_list:
                                if result.get('Prompt Name') == prompt_name:
                                    previous_value = result.get('Extracted Value', '')
                                    break
                            
                            # Customize prompt based on previous result
                            if previous_value and previous_value.lower() in ["none", "null", "n/a", "na"]:
                                enhanced_instruction = f"{pdf_instruction}\n\nCRITICAL: Previous extraction returned '{previous_value}'. This may be incorrect. Please be extremely thorough and look for ANY mention of this attribute, even if it's not explicitly labeled. Consider technical specifications, material properties, dimensions, or any related information that might indicate this attribute's value."
                            else:
                                enhanced_instruction = f"{pdf_instruction}\n\nIMPORTANT: This is a final recheck. Be more thorough and consider alternative interpretations. If the information is not explicitly stated, try to infer from related context or technical specifications."
                            
                            enhanced_pdf_input = {
                                "context": context_text,
                                "extraction_instructions": enhanced_instruction,
                                "attribute_key": attribute_key,
                                "part_number": part_number if part_number else "Not Provided"
                            }
                            
                            debug_logger.llm_request(
                                f"Final fallback extraction for {attribute_key}",
                                "pdf_chain",
                                0.7,
                                1500,  # Increased token limit for more thorough analysis
                                context={"step": "stage3_llm_request", "attribute": attribute_key}
                            )
                            
                            json_result_str = loop.run_until_complete(
                                _invoke_chain_and_process(st.session_state.pdf_chain, enhanced_pdf_input, f"{attribute_key} (Final Fallback)")
                            )
                            run_time = time.time() - start_time
                            
                            debug_logger.llm_response(
                                "pdf_chain",
                                json_result_str if json_result_str else "",
                                len(json_result_str) if json_result_str else 0,
                                run_time,
                                context={"step": "stage3_llm_response", "attribute": attribute_key}
                            )
                            
                            time.sleep(SLEEP_INTERVAL_SECONDS)
                            
                        except Exception as e:
                            logger.error(f"Error during Stage 3 (Final Fallback) call for '{attribute_key}': {e}", exc_info=True)
                            json_result_str = f'{{"error": "Exception during Stage 3 call: {e}"}}'
                            run_time = time.time() - start_time
                            
                            debug_logger.exception(e, context={
                                "step": "stage3_exception",
                                "attribute": attribute_key,
                                "duration": run_time
                            })
                
                # Parse final fallback result
                final_answer_value = "Error"
                parse_error = None
                is_rate_limit = False
                raw_output = json_result_str if json_result_str else '{"error": "Stage 3 did not run"}'
                
                try:
                    string_to_parse = raw_output.strip()
                    parsed_json = extract_json_from_string(string_to_parse)
                    
                    debug_logger.data_transformation(
                        f"Final fallback JSON parsing for {attribute_key}",
                        string_to_parse,
                        parsed_json,
                        context={"step": "stage3_json_parsing", "attribute": attribute_key}
                    )
                    
                    if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                        parsed_value = str(parsed_json[attribute_key])
                        # For final fallback, be more lenient with empty values and "none" responses
                        if (parsed_value.strip() == "" or 
                            "not found" in parsed_value.lower() or
                            parsed_value.lower() in ["none", "null", "n/a", "na"]):
                            final_answer_value = "NOT FOUND (Final)"
                        else:
                            final_answer_value = parsed_value
                            logger.success(f"Stage 3 successful for '{attribute_key}' with value: {parsed_value}")
                    elif isinstance(parsed_json, dict) and "error" in parsed_json:
                        final_answer_value = f"Error: {parsed_json['error'][:100]}"
                        parse_error = ValueError(f"Stage 3 Error: {parsed_json['error']}")
                    else:
                        final_answer_value = "Unexpected JSON Format (Final)"
                        parse_error = ValueError(f"Stage 3 Unexpected JSON format")
                        
                except Exception as processing_exc:
                    parse_error = processing_exc
                    final_answer_value = "Processing Error (Final)"
                    logger.error(f"Error processing Stage 3 result for '{attribute_key}'. Error: {processing_exc}")
                    
                    debug_logger.exception(processing_exc, context={
                        "step": "stage3_processing_exception",
                        "attribute": attribute_key,
                        "raw_output": string_to_parse
                    })
                
                # Update the result in the list
                for i, result in enumerate(extraction_results_list):
                    if result.get('Prompt Name') == prompt_name:
                        # Calculate total latency including previous stages
                        previous_latency = result.get('Latency (s)', 0.0)
                        total_latency = previous_latency + round(run_time, 2)
                        
                        # Check if we should preserve the original value (rollback logic)
                        original_value = result.get('Extracted Value', '')
                        original_source = result.get('Source', 'Unknown')
                        
                        # Rollback conditions: preserve original value if Stage 3 failed
                        should_rollback = (
                            # Preserve "none" values when confirmed by recheck
                            (original_value.lower() in ["none", "null", "n/a", "na"] and final_answer_value == "NOT FOUND (Final)") or
                            # Rollback to original when Stage 3 has errors
                            bool(parse_error) or
                            final_answer_value in ["Error", "Processing Error (Final)", "Unexpected JSON Format (Final)"]
                        )
                        
                        # Determine final value
                        if should_rollback:
                            final_display_value = original_value  # Keep original value
                            final_source = original_source  # Keep original source
                            is_success = result.get('Is Success', False)  # Keep original success status
                            is_not_found = result.get('Is Not Found', False)  # Keep original not found status
                            is_error = result.get('Is Error', False)  # Keep original error status
                        else:
                            final_display_value = final_answer_value
                            final_source = source
                            is_success = not bool(parse_error) and final_answer_value not in ["NOT FOUND (Final)", "Error", "Processing Error (Final)", "Unexpected JSON Format (Final)"]
                            is_not_found = final_answer_value in ["NOT FOUND (Final)"]
                            is_error = bool(parse_error)
                        
                        # Update the result
                        extraction_results_list[i].update({
                            'Extracted Value': final_display_value,
                            'Source': final_source,
                            'Raw Output': raw_output if not should_rollback else result.get('Raw Output', raw_output),
                            'Parse Error': str(parse_error) if parse_error and not should_rollback else result.get('Parse Error'),
                            'Is Success': is_success,
                            'Is Error': is_error,
                            'Is Not Found': is_not_found,
                            'Is Rate Limit': is_rate_limit,
                            'Latency (s)': total_latency
                        })
                        
                        debug_logger.info(f"Stage 3 result updated for {attribute_key}", data={
                            "final_value": final_display_value,
                            "original_value": original_value,
                            "should_rollback": should_rollback,
                            "total_latency": total_latency,
                            "success": is_success
                        }, context={"step": "stage3_result_updated", "attribute": attribute_key})
                        
                        # Show feedback for rollback
                        if should_rollback and bool(parse_error):
                            logger.info(f"Stage 3: Rolled back to original '{original_value}' for '{attribute_key}' (Stage 3 error: {parse_error})")
                        elif should_rollback and original_value.lower() in ["none", "null", "n/a", "na"]:
                            logger.info(f"Stage 3: Preserved original '{original_value}' for '{attribute_key}' (confirmed by recheck)")
                        break
        else:
            st.success("No attributes need final fallback - all extractions completed successfully.")
            debug_logger.info("No final fallback needed", data={
                "reason": "All attributes successful in previous stages"
            }, context={"step": "stage3_skipped_all_successful"})
        
        # --- Stage Summary ---
        # (REMOVED Extraction Stage Summary UI section)
        extraction_successful = True # Assume success unless critical errors occurred (e.g., chain init)

        if extraction_successful:
            st.session_state.evaluation_results = extraction_results_list
            st.session_state.extraction_performed = True
            st.session_state.extraction_attempts = 0  # Reset counter on success
            logger.info("Extraction completed successfully, setting extraction_performed=True")
            st.success("Extraction complete (3-stage process: Web → NuMind → Final Fallback).")
            debug_logger.session_state("evaluation_results", extraction_results_list, context={"step": "results_stored"})
            debug_logger.session_state("extraction_performed", True, context={"step": "extraction_flag_set"})
            debug_logger.session_state("extraction_attempts", 0, context={"step": "attempts_reset"})
            debug_logger.info("Extraction completed successfully", context={"step": "extraction_success"})
        else:
            st.error("Extraction process encountered critical issues.")
            st.session_state.evaluation_results = extraction_results_list
            st.session_state.extraction_performed = True
            st.session_state.extraction_attempts = 0  # Reset counter even on error
            logger.info("Extraction completed with issues, setting extraction_performed=True")
            debug_logger.warning("Extraction completed with issues", data={
                "results_count": len(extraction_results_list)
            }, context={"step": "extraction_with_issues"})

        # --- Card-based UI for Extracted Attributes ---
        # (REMOVED 3. Enter Ground Truth & Evaluate and 4. View Raw Mistral Extraction UI sections)
        import re
        def strip_html_tags(text):
            if not isinstance(text, str):
                return text
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)

        # --- Update CSS for compact cards ---
        st.markdown(
            """
            <style>
            .attribute-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border: 1.5px solid #1e3c72;
                border-radius: 6px;
                padding: 0.4rem 0.6rem 0.6rem 0.6rem;
                min-width: 0;
                width: 100%;
                box-shadow: 0 2px 6px rgba(30, 60, 114, 0.07);
                margin: 0 0 0.5rem 0;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }
            .attribute-card h4 {
                color: #1e3c72;
                margin: 0 0 0.3rem 0;
                font-size: 1em;
                font-weight: 700;
                border-bottom: 1px solid #1e3c72;
                padding-bottom: 0.2rem;
                width: 100%;
            }
            .attribute-value {
                background: #fff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                margin: 0.1rem 0 0 0;
                font-weight: 500;
                font-size: 0.95em;
                width: 100%;
                box-sizing: border-box;
            }
            /* Remove extra margin between columns */
            .element-container .stColumn > div {
                /* Removed forced margin, revert to Streamlit default */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.divider()
        st.subheader("🗂️ Extracted Attributes (Compact Grid)")
        # Display cards in a responsive grid: up to 4 per row, only name and value
        results = [r for r in st.session_state.evaluation_results if isinstance(r, dict)]
        num_cols = 5
        for i in range(0, len(results), num_cols):
            row_results = results[i:i+num_cols]
            cols = st.columns(min(len(row_results), num_cols))
            for j, result in enumerate(row_results):
                with cols[j]:
                    attr_name = result.get('Prompt Name', 'Unknown Attribute')
                    value = result.get('Extracted Value', '')
                    st.markdown(f'''<div class=\"attribute-card\">
                        <h4>{attr_name}</h4>
                        <div class=\"attribute-value\">{value}</div>
                    </div>''', unsafe_allow_html=True)

        st.divider()

        # --- Main Extraction UI ---
        if st.session_state.get("evaluation_results"):
            # --- Manual Recheck Section ---
            st.divider()
            st.subheader("🔄 Manual Attribute Recheck")
            
            with st.expander("ℹ️ How to use Manual Recheck"):
                st.markdown("""
                **Manual Recheck allows you to re-extract specific attributes that may have been missed:**
                1. **Select attributes** from the dropdown below
                2. **Click "Run Manual Recheck"** button
                3. **Wait for results** - each attribute will be rechecked with enhanced prompts
                4. **Review results** - successful extractions will show green checkmarks
                **When to use:**
                - Attributes that returned "NOT FOUND"
                - Attributes that returned "none" (might be incorrect)
                - Attributes with errors or unexpected formats
                - Any attribute you suspect might have been missed
                **What happens:**
                - Uses more document chunks (15 instead of 8-12)
                - Enhanced prompts specifically for rechecking
                - Preserves original "none" values if recheck confirms they're correct
                """)
            manual_recheck_candidates = []
            for result in st.session_state.evaluation_results:
                if isinstance(result, dict):
                    manual_recheck_candidates.append(result.get('Prompt Name', ''))
            if manual_recheck_candidates:
                none_candidates = []
                other_candidates = []
                for result in st.session_state.evaluation_results:
                    if isinstance(result, dict) and result.get('Prompt Name') in manual_recheck_candidates:
                        extracted_value = result.get('Extracted Value', '')
                        if extracted_value and extracted_value.lower() in ["none", "null", "n/a", "na"]:
                            none_candidates.append(result.get('Prompt Name'))
                        else:
                            other_candidates.append(result.get('Prompt Name'))
                selected_for_recheck = st.multiselect(
                    "Select attributes to recheck:",
                    options=manual_recheck_candidates,
                    default=manual_recheck_candidates[:3],
                    help="Select any attribute to re-extract from the PDF using the RAG LLM. Useful for double-checking or improving any extraction, not just failed ones.",
                    key="manual_recheck_multiselect"
                )
                part_number = st.session_state.get("part_number_input", "").strip()
                # Use a unique key for the button
                manual_recheck_clicked = st.button("🔄 Run Manual Recheck", type="primary", key="manual_recheck_button")
                if selected_for_recheck and manual_recheck_clicked:
                    st.info(f"Running manual recheck for {len(selected_for_recheck)} selected attributes...")
                    # TODO: Implement the actual manual recheck logic here.
                    # For now, just display a placeholder message and do not modify st.session_state.evaluation_results.
                    st.warning("Manual recheck logic not yet implemented. Results will update here when available.")
                elif not selected_for_recheck:
                    st.info("Select at least one attribute to enable manual recheck.")
                # Always display the last results from session state below
            else:
                st.success("All attributes have been successfully extracted! No manual recheck needed.")
            # --- Export Section ---
            st.divider()
            st.header("6. Export Results")
            if st.session_state.evaluation_results:
                export_df = pd.DataFrame(st.session_state.evaluation_results)
                export_summary = st.session_state.evaluation_metrics if st.session_state.evaluation_metrics else {}
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                csv_data = convert_df_to_csv(export_df)
                json_summary_data = json.dumps(export_summary, indent=2).encode('utf-8')
                export_cols = st.columns(2)
                with export_cols[0]:
                    st.download_button(
                        label="📥 Download Detailed Results (CSV)",
                        data=csv_data,
                        file_name='detailed_extraction_results.csv',
                        mime='text/csv',
                        key='download_csv'
                    )
                with export_cols[1]:
                    st.download_button(
                        label="📥 Download Summary Metrics (JSON)",
                        data=json_summary_data,
                        file_name='evaluation_summary.json',
                        mime='application/json',
                        key='download_json'
                    )
            else:
                st.info("Process documents and calculate metrics to enable export.")
        else:
            st.warning("Extraction process completed, but no valid results were generated for some fields. Check logs or raw outputs if available.")

    # --- Block 3: Handle cases where extraction ran but yielded nothing, or hasn't run ---
    # This logic might need review depending on how Stage 1/2 errors are handled
    elif (st.session_state.pdf_chain or st.session_state.web_chain) and st.session_state.extraction_performed:
        st.warning("Extraction process completed, but no valid results were generated for some fields. Check logs or raw outputs if available.")


    