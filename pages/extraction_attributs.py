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

import streamlit as st
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
        padding: 0.7rem 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
    }
    
    .header-band h1 {
        font-size: 2.2em;
        margin: 0;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-band h2 {
        font-size: 1.8em;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Button styling with blue theme */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #4a90e2 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Section headers styling */
    .section-header {
        color: #1e3c72;
        font-size: 2em;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Info boxes styling */
    .stAlert {
        border-left: 4px solid #1e3c72;
    }
    
    /* Success messages styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #1e3c72;
    }
    
    /* Warning messages styling */
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #1e3c72;
    }
    
    /* Horizontal table styling */
    .horizontal-table {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .attribute-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #1e3c72;
        border-radius: 12px;
        padding: 1rem;
        min-width: 300px;
        flex: 1;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
        transition: all 0.3s ease;
    }
    
    .attribute-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.2);
    }
    
    .attribute-card h4 {
        color: #1e3c72;
        margin: 0 0 0.5rem 0;
        font-size: 1.1em;
        font-weight: 600;
        border-bottom: 2px solid #1e3c72;
        padding-bottom: 0.5rem;
    }
    
    .attribute-value {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .attribute-source {
        font-size: 0.8em;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    .success-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .success-true {
        background-color: #28a745;
    }
    
    .success-false {
        background-color: #dc3545;
    }
    
    /* Data editor styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #1e3c72;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #1e3c72;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
    }
    
    /* Right pane styling */
    .right-pane {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 3px solid #1e3c72;
        border-radius: 0 15px 15px 0;
        padding: 1.5rem;
        box-shadow: -5px 0 15px rgba(30, 60, 114, 0.1);
        max-height: 90vh;
        overflow-y: auto;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
        margin-bottom: 1rem;
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
def initialize_chatbot():
    """Initialize chatbot components"""
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        groq_client = Groq(api_key=GROQ_API_KEY)
        return groq_client
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None

def get_chat_response(groq_client, user_message, extraction_data):
    """Get response from Groq chatbot based on extraction data"""
    try:
        # Create context from extraction data
        context = "Extracted data:\n"
        if extraction_data:
            for item in extraction_data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key not in ['Raw Output', 'Parse Error', 'Is Success', 'Is Error', 'Is Not Found', 'Is Rate Limit', 'Latency (s)', 'Exact Match', 'Case-Insensitive Match']:
                            if value and value != 'NOT FOUND' and value != 'ERROR':
                                context += f"{key}: {value}\n"
        
        if not context.strip() or context.strip() == "Extracted data:":
            return "I don't have any extracted data to work with yet. Please complete the extraction process first."
        
        prompt = f"""You are a helpful assistant for LEONI parts data. You have access to the following extracted information:

{context}

User question: {user_message}

Please provide a helpful and accurate response based on the extracted data. If the information is not available in the extracted data, please say so clearly. Be concise but informative."""
        
        response = groq_client.chat.completions.create(
            model="qwen-qwq-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return f"Sorry, I encountered an error while processing your request. Please try again."

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
st.markdown(
    """
    <div class="header-band">
        <h1>LEONI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Blue band header with LEONI
st.markdown("""
    <div class="header-band">
        <h1>LEOPARTS</h1>
        <h2>LEONI</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📄 PDF Attribute Extraction")
st.markdown("Upload your PDF documents and automatically extract key attributes.")

if not config.GROQ_API_KEY:
    st.warning("Groq API Key not found. Please set the GROQ_API_KEY environment variable.", icon="⚠️")

# --- Stepper UI (place at the top of your main page) ---
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = 'upload'  # 'upload', 'processing', 'results'
if 'stepper_stage' not in st.session_state:
    st.session_state.stepper_stage = 0  # 0: upload, 1: processing, 2: results

st.markdown("""
<style>
.stepper {{ display: flex; justify-content: center; margin: 2rem 0; }}
.step {{ display: flex; align-items: center; }}
.step-icon {{
    width: 36px; height: 36px; border-radius: 50%; background: #e0e0e0;
    color: #1e3c72; display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 1.2em; border: 2px solid #1e3c72;
    transition: background 0.3s, color 0.3s;
}}
.step-icon.active {{ background: #1e3c72; color: #fff; }}
.step-icon.done {{ background: #28a745; color: #fff; border-color: #28a745; }}
.step-label {{ margin: 0 1rem; font-weight: 500; color: #1e3c72; }}
.step-connector {{
    width: 40px; height: 2px; background: #1e3c72; margin: 0 0.5rem;
}}
</style>
<div class="stepper">
    <div class="step">
        <div class="step-icon {upload_done}">1</div>
        <div class="step-label">Upload</div>
        <div class="step-connector"></div>
    </div>
    <div class="step">
        <div class="step-icon {processing_active}">2</div>
        <div class="step-label">Processing</div>
        <div class="step-connector"></div>
    </div>
    <div class="step">
        <div class="step-icon {results_active}">3</div>
        <div class="step-label">Results</div>
    </div>
</div>
""".format(
    upload_done="done" if st.session_state.stepper_stage > 0 else "active" if st.session_state.stepper_stage == 0 else "",
    processing_active="done" if st.session_state.stepper_stage > 1 else "active" if st.session_state.stepper_stage == 1 else "",
    results_active="active" if st.session_state.stepper_stage == 2 else ""
), unsafe_allow_html=True)

# --- Step 1: File Upload (Main Page) ---
if st.session_state.processing_state == 'upload':
    st.markdown("""
        <div style="text-align:center; margin-top:2rem;">
            <h2>Start by uploading your PDF document</h2>
            <p>We'll help you extract all the important information in a few easy steps!</p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload your PDF file(s)",
        type="pdf",
        accept_multiple_files=True,
        key="main_pdf_uploader"
    )
    st.text_input("Part Number (optional)", key="part_number_input", placeholder="Enter part number if you know it")
    process_button = st.button("Process Document(s)", type="primary")
    if process_button and uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.processing_state = 'processing'
        st.session_state.stepper_stage = 1
        st.rerun()
    elif process_button and not uploaded_files:
        st.warning("Please upload at least one PDF file to continue.")

# --- Step 2: Processing Stepper ---
elif st.session_state.processing_state == 'processing':
    st.info("Processing your documents. Please wait...")
    # Here, run your extraction logic (Web, NuMind, Fallback)
    # As each sub-stage completes, you can update a sub-step variable if desired
    # When all done:
    # st.session_state.processing_state = 'results'
    # st.session_state.stepper_stage = 2
    # st.rerun()

# --- Step 3: Results ---
elif st.session_state.processing_state == 'results':
    st.success("Extraction complete! See your results below.")
    # Show results cards/table here

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


# --- Main Layout ---
# Remove the left_col, right_col = st.columns([2, 1]) and any 'with left_col:' or 'with right_col:'

# After extraction is complete and before the export/download section:
if st.session_state.evaluation_results:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; 
                    padding: 1rem; 
                    border-radius: 15px; 
                    text-align: center; 
                    margin-bottom: 1rem;">
            <h3 style="margin: 0; font-size: 1.5em;">📊 Extraction Results</h3>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('''
    <style>
    .horizontal-attributes-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin: 1.5rem 0;
        justify-content: flex-start;
        align-items: flex-start;
    }
    .attribute-card-h {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #1e3c72;
        border-radius: 12px;
        padding: 1rem;
        width: 320px;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .attribute-card-h:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.2);
    }
    .attribute-card-h h4 {
        color: #1e3c72;
        margin: 0 0 0.5rem 0;
        font-size: 1.1em;
        font-weight: 600;
        border-bottom: 2px solid #1e3c72;
        padding-bottom: 0.5rem;
    }
    .attribute-value-h {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    </style>
    <div class="horizontal-attributes-container">
    ''', unsafe_allow_html=True)
    extracted_data = {}
    for result in st.session_state.evaluation_results:
        if isinstance(result, dict):
            prompt_name = result.get('Prompt Name', 'Unknown')
            extracted_value = result.get('Extracted Value', '')
            if extracted_value and extracted_value != 'NOT FOUND' and extracted_value != 'ERROR':
                extracted_data[prompt_name] = extracted_value
    for key, value in extracted_data.items():
        display_value = value[:100] + "..." if len(value) > 100 else value
        st.markdown(f'''
            <div class="attribute-card-h">
                <h4>🔍 {key}</h4>
                <div class="attribute-value-h" title="{value}">{display_value}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ... then the Export Results section comes after ...

    # --- Export Section --- 
    st.divider()
    st.header("6. Export Results")

    if st.session_state.evaluation_results:
        # Prepare data for export
        export_df = pd.DataFrame(st.session_state.evaluation_results)
        export_summary = st.session_state.evaluation_metrics if st.session_state.evaluation_metrics else {}

        # Convert DataFrame to CSV
        @st.cache_data # Cache the conversion
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(export_df)

        # Convert summary dict to JSON
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

# --- Block 3: Handle cases where extraction ran but yielded nothing, or hasn't run ---
# This logic might need review depending on how Stage 1/2 errors are handled
if (st.session_state.pdf_chain or st.session_state.web_chain) and st.session_state.extraction_performed and not st.session_state.evaluation_results:
    st.warning("Extraction process completed, but no valid results were generated for some fields. Check logs or raw outputs if available.")

    # --- Remove the two-column layout and right_col ---
    # (Remove: left_col, right_col = st.columns([2, 1]) and all 'with right_col:' blocks)

    # ... existing code ...

    # After the extraction results are available and after the stepper/results section:
    if st.session_state.evaluation_results:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        color: white; 
                        padding: 1rem; 
                        border-radius: 15px; 
                        text-align: center; 
                        margin-bottom: 1rem;">
                <h3 style="margin: 0; font-size: 1.5em;">📊 Extraction Results</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Display extraction results in a beautiful horizontal flexbox format
        st.markdown('''
        <style>
        .horizontal-attributes-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin: 1.5rem 0;
            justify-content: flex-start;
            align-items: flex-start;
        }
        .attribute-card-h {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #1e3c72;
            border-radius: 12px;
            padding: 1rem;
            width: 320px;
            box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }
        .attribute-card-h:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(30, 60, 114, 0.2);
        }
        .attribute-card-h h4 {
            color: #1e3c72;
            margin: 0 0 0.5rem 0;
            font-size: 1.1em;
            font-weight: 600;
            border-bottom: 2px solid #1e3c72;
            padding-bottom: 0.5rem;
        }
        .attribute-value-h {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 0.5rem;
            margin: 0.5rem 0;
            font-weight: 500;
        }
        </style>
        <div class="horizontal-attributes-container">
        ''', unsafe_allow_html=True)
        
        # Create a summary of extracted data
        extracted_data = {}
        for result in st.session_state.evaluation_results:
            if isinstance(result, dict):
                prompt_name = result.get('Prompt Name', 'Unknown')
                extracted_value = result.get('Extracted Value', '')
                if extracted_value and extracted_value != 'NOT FOUND' and extracted_value != 'ERROR':
                    extracted_data[prompt_name] = extracted_value
        
        # Display each extracted item as a horizontal card
        for key, value in extracted_data.items():
            display_value = value[:100] + "..." if len(value) > 100 else value
            st.markdown(f'''
                <div class="attribute-card-h">
                    <h4>🔍 {key}</h4>
                    <div class="attribute-value-h" title="{value}">{display_value}</div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # ... existing code ...
    
    # Chatbot Section
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; 
                    padding: 1rem; 
                    border-radius: 15px; 
                    text-align: center; 
                    margin: 2rem 0 1rem 0;">
            <h3 style="margin: 0; font-size: 1.5em;">💬 Chat with Your Data</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                    <div class="chat-message user">
                        <strong>You:</strong> {message['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant">
                        <strong>Assistant:</strong> {message['content']}
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if groq_client and st.session_state.evaluation_results:
        user_message = st.text_input("Ask about your extracted data:", key="chat_input", placeholder="e.g., What is the part number?")
        
        if st.button("Send", key="send_chat"):
            if user_message.strip():
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_message
                })
                
                # Get response from chatbot
                with st.spinner("Thinking..."):
                    response = get_chat_response(groq_client, user_message, st.session_state.evaluation_results)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                # Clear input and rerun to show new messages
                st.rerun()
    elif not groq_client:
        st.warning("⚠️ Chatbot not available - API key missing")
    elif not st.session_state.evaluation_results:
        st.info("💬 Chat will be available once you extract data")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    