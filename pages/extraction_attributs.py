import asyncio
import json
import os
import re
import subprocess
import sys
import time
from typing import List

import nest_asyncio
import pandas as pd
import requests
import streamlit as st
from groq import Groq
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import GROQ_API_KEY
from debug_logger import debug_logger, DebugTimer, log_streamlit_state, log_json_parsing
from extraction_prompts import (
    MATERIAL_PROMPT, MATERIAL_NAME_PROMPT, PULL_TO_SEAT_PROMPT, GENDER_PROMPT,
    HEIGHT_MM_PROMPT, LENGTH_MM_PROMPT, WIDTH_MM_PROMPT, NUMBER_OF_CAVITIES_PROMPT,
    NUMBER_OF_ROWS_PROMPT, MECHANICAL_CODING_PROMPT, COLOUR_PROMPT, COLOUR_CODING_PROMPT,
    WORKING_TEMPERATURE_PROMPT, HOUSING_SEAL_PROMPT, WIRE_SEAL_PROMPT, SEALING_PROMPT,
    SEALING_CLASS_PROMPT, CONTACT_SYSTEMS_PROMPT, TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT, CLOSED_CAVITIES_PROMPT, PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT, SET_KIT_PROMPT, HV_QUALIFIED_PROMPT
)
from extraction_prompts_web import (
    MATERIAL_FILLING_WEB_PROMPT, MATERIAL_NAME_WEB_PROMPT, PULL_TO_SEAT_WEB_PROMPT,
    GENDER_WEB_PROMPT, HEIGHT_MM_WEB_PROMPT, LENGTH_MM_WEB_PROMPT, WIDTH_MM_WEB_PROMPT,
    NUMBER_OF_CAVITIES_WEB_PROMPT, NUMBER_OF_ROWS_WEB_PROMPT, MECHANICAL_CODING_WEB_PROMPT,
    COLOUR_WEB_PROMPT, COLOUR_CODING_WEB_PROMPT, MAX_WORKING_TEMPERATURE_WEB_PROMPT,
    MIN_WORKING_TEMPERATURE_WEB_PROMPT, HOUSING_SEAL_WEB_PROMPT, WIRE_SEAL_WEB_PROMPT,
    SEALING_WEB_PROMPT, SEALING_CLASS_WEB_PROMPT, CONTACT_SYSTEMS_WEB_PROMPT,
    TERMINAL_POSITION_ASSURANCE_WEB_PROMPT, CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT,
    CLOSED_CAVITIES_WEB_PROMPT, PRE_ASSEMBLED_WEB_PROMPT, CONNECTOR_TYPE_WEB_PROMPT,
    SET_KIT_WEB_PROMPT, HV_QUALIFIED_WEB_PROMPT
)
from llm_interface import (
    initialize_llm, create_pdf_extraction_chain, create_web_extraction_chain,
    create_numind_extraction_chain, scrape_website_table_html, _invoke_chain_and_process,
    extract_with_numind_using_schema, extract_specific_attribute_from_numind_result,
    get_default_extraction_schema
)
from numind_schema_config import get_custom_schema, get_custom_instructions
from pdf_processor import process_uploaded_pdfs, fetch_chunks
from vector_store import get_embedding_function, setup_vector_store

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.debug("Logger initialized with DEBUG level.")

# Force Python to use pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Define prompts dictionary
prompts_to_run = {
    "Material Filling": {"pdf": MATERIAL_PROMPT, "web": MATERIAL_FILLING_WEB_PROMPT},
    "Material Name": {"pdf": MATERIAL_NAME_PROMPT, "web": MATERIAL_NAME_WEB_PROMPT},
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
    "Max. Working Temperature [°C]": {"pdf": WORKING_TEMPERATURE_PROMPT, "web": MAX_WORKING_TEMPERATURE_WEB_PROMPT},
    "Min. Working Temperature [°C]": {"pdf": WORKING_TEMPERATURE_PROMPT, "web": MIN_WORKING_TEMPERATURE_WEB_PROMPT},
    "Housing Seal": {"pdf": HOUSING_SEAL_PROMPT, "web": HOUSING_SEAL_WEB_PROMPT},
    "Wire Seal": {"pdf": WIRE_SEAL_PROMPT, "web": WIRE_SEAL_WEB_PROMPT},
    "Sealing": {"pdf": SEALING_PROMPT, "web": SEALING_WEB_PROMPT},
    "Sealing Class": {"pdf": SEALING_CLASS_PROMPT, "web": SEALING_CLASS_WEB_PROMPT},
    "Contact Systems": {"pdf": CONTACT_SYSTEMS_PROMPT, "web": CONTACT_SYSTEMS_WEB_PROMPT},
    "Terminal Position Assurance": {"pdf": TERMINAL_POSITION_ASSURANCE_PROMPT, "web": TERMINAL_POSITION_ASSURANCE_WEB_PROMPT},
    "Connector Position Assurance": {"pdf": CONNECTOR_POSITION_ASSURANCE_PROMPT, "web": CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT},
    "Name Of Closed Cavities": {"pdf": CLOSED_CAVITIES_PROMPT, "web": CLOSED_CAVITIES_WEB_PROMPT},
    "Pre-assembled": {"pdf": PRE_ASSEMBLED_PROMPT, "web": PRE_ASSEMBLED_WEB_PROMPT},
    "Type Of Connector": {"pdf": CONNECTOR_TYPE_PROMPT, "web": CONNECTOR_TYPE_WEB_PROMPT},
    "Set/Kit": {"pdf": SET_KIT_PROMPT, "web": SET_KIT_WEB_PROMPT},
    "HV Qualified": {"pdf": HV_QUALIFIED_PROMPT, "web": HV_QUALIFIED_WEB_PROMPT}
}

# CSS styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    [data-testid='stSidebarNav'] {display: none;}
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
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    .section-header {
        color: #1e3c72;
        font-size: 2em;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .stAlert {
        border-left: 4px solid #1e3c72;
    }
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #1e3c72;
    }
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #1e3c72;
    }
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
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);
    }
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
    .right-pane {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 3px solid #1e3c72;
        border-radius: 0 15px 15px 0;
        padding: 1.5rem;
        box-shadow: -5px 0 15px rgba(30, 60, 114, 0.1);
        max-height: 90vh;
        overflow-y: auto;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: white;
        border-radius: 15px;
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
        display: flex;
        flex-wrap: nowrap;
        gap: 1rem;
        justify-content: flex-start;
        align-items: stretch;
        margin-bottom: 1rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }
    .result-item {
        flex: 0 0 260px;
        min-width: 220px;
        max-width: 260px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
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
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
def initialize_session_state():
    defaults = {
        'retriever': None,
        'pdf_chain': None,
        'web_chain': None,
        'numind_chain': None,
        'processed_files': [],
        'evaluation_results': [],
        'evaluation_metrics': None,
        'extraction_performed': False,
        'extraction_attempts': 0,
        'scraped_table_html_cache': None,
        'current_part_number_scraped': None,
        'processed_documents': [],
        'uploaded_file_data': [],
        'show_raw_extraction': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

def reset_evaluation_state():
    """Reset evaluation-related session state variables."""
    st.session_state.evaluation_results = []
    st.session_state.evaluation_metrics = None
    st.session_state.extraction_performed = False
    st.session_state.scraped_table_html_cache = None
    st.session_state.current_part_number_scraped = None
    st.session_state.processed_documents = []
    st.session_state.uploaded_file_data = []

# Cache resources
@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings function."""
    try:
        embeddings = get_embedding_function()
        logger.success("Embedding function initialized.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize embedding model. Error: {e}")
        raise

@st.cache_resource
def initialize_llm_cached():
    """Initialize LLM function."""
    try:
        llm_instance = initialize_llm()
        logger.success("LLM initialized.")
        return llm_instance
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize LLM. Error: {e}")
        raise

# Initialize core components
try:
    embedding_function = initialize_embeddings()
    llm = initialize_llm_cached()
except Exception:
    st.error("Core components failed to initialize. Cannot continue.")
    st.stop()

def install_playwright_browsers():
    """Install Playwright browsers if needed."""
    if 'playwright_installed' not in st.session_state:
        logger.info("Installing Playwright browsers...")
        try:
            process = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                                    capture_output=True, text=True, check=False)
            if process.returncode == 0:
                logger.success("Playwright browsers installed.")
            else:
                logger.error(f"Playwright install failed: {process.stderr}")
        except Exception as e:
            logger.error(f"Playwright installation error: {e}", exc_info=True)
            st.warning(f"Playwright installation failed: {e}")
        st.session_state.playwright_installed = True

install_playwright_browsers()

def render_extraction_progress(stage1_count, stage2_count, stage3_count, numind_time=None, none_responses=None):
    """Render extraction progress UI."""
    st.info(f"Running Stage 1 (Web Data Extraction) for {stage1_count} attributes...")
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    with progress_col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        color: white; padding: 0.5rem; border-radius: 10px; text-align: center;">
                <strong>Stage 1: Web</strong><br><small>Web scraping & extraction</small>
            </div>
        """, unsafe_allow_html=True)
    with progress_col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                        color: white; padding: 0.5rem; border-radius: 10px; text-align: center;">
                <strong>Stage 2: NuMind</strong><br><small>Structured extraction</small>
            </div>
        """, unsafe_allow_html=True)
    with progress_col3:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%); 
                        color: white; padding: 0.5rem; border-radius: 10px; text-align: center;">
                <strong>Stage 3: Fallback</strong><br><small>Final recheck</small>
            </div>
        """, unsafe_allow_html=True)
    st.info(f"Running Stage 2 (NuMind Fallback) for {stage2_count} attributes...")
    st.success("Using NuMind for structured extraction...")
    if numind_time is not None:
        st.success(f"NuMind extraction completed in {numind_time:.2f} seconds.")
    st.info(f"Running Stage 3 (Final Fallback) for {stage3_count} attributes...")
    if none_responses:
        st.warning(f"⚠️ Including {len(none_responses)} attributes with 'none' responses.")

def initialize_chatbot():
    """Initialize chatbot components."""
    try:
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        return groq_client
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None

def get_chat_response(groq_client, user_message, extraction_data):
    """Get response from Groq chatbot based on extraction data."""
    try:
        context = "Extracted data:\n"
        if extraction_data:
            for item in extraction_data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key not in ['Raw Output', 'Parse Error', 'Is Success', 'Is Error', 'Is Not Found', 'Is Rate Limit', 'Latency (s)', 'Exact Match', 'Case-Insensitive Match']:
                            if value and value != 'NOT FOUND' and value != 'ERROR':
                                context += f"{key}: {value}\n"
        if context.strip() == "Extracted data:":
            return "No extracted data available. Please complete the extraction process."
        
        prompt = f"""You are a helpful assistant for LEONI parts data. Extracted information:
{context}
User question: {user_message}
Provide a helpful, accurate response based on the extracted data. If the information is unavailable, state so clearly."""
        
        response = groq_client.chat.completions.create(
            model="qwen-qwq-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return "Error processing your request. Please try again."

def extract_json_from_string(s):
    """Extract the first valid JSON object from a string."""
    if not s or not isinstance(s, str):
        return None
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    match = re.search(r'\{.*\}', s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

# Navigation Sidebar
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

# UI Layout
st.markdown("""
    <div class="header-band">
        <h1>LEOPARTS</h1>
        <h2>LEONI</h2>
    </div>
""", unsafe_allow_html=True)

# Initialize chatbot
groq_client = initialize_chatbot()

# Main layout
st.markdown("### 📄 PDF Attribute Extraction")
st.markdown("Upload PDF documents to extract key attributes.")

if not GROQ_API_KEY:
    st.warning("Groq API Key not found. Set the GROQ_API_KEY environment variable.", icon="⚠️")

# Sidebar for PDF Upload and Processing
with st.sidebar:
    st.header("1. Document Processing")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    st.text_input("Enter Part Number (Optional):", key="part_number_input", value=st.session_state.get("part_number_input", ""))
    process_button = st.button("Process Uploaded Documents", key="process_button", type="primary")

    st.subheader("Processing Status")
    if st.session_state.pdf_chain and st.session_state.web_chain and st.session_state.processed_files:
        st.success(f"Ready. Processed: {', '.join(st.session_state.processed_files)}")
    else:
        st.info("Upload and process PDF documents to view extracted data.")

# Process uploaded files
if process_button and uploaded_files:
    if not embedding_function or not llm:
        st.error("Core components failed to initialize.")
    else:
        reset_evaluation_state()
        st.session_state.retriever = None
        st.session_state.pdf_chain = None
        st.session_state.web_chain = None
        st.session_state.processed_files = []
        
        filenames = [f.name for f in uploaded_files]
        st.session_state.uploaded_file_data = [(f.name, f.getvalue()) for f in uploaded_files]
        logger.info(f"Processing {len(filenames)} files: {', '.join(filenames)}")

        with st.spinner("Processing PDFs..."):
            try:
                start_time = time.time()
                temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                loop = asyncio.get_event_loop()
                processed_docs = loop.run_until_complete(process_uploaded_pdfs(uploaded_files, temp_dir))
                logger.info(f"PDF processing took {time.time() - start_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"PDF processing error: {e}", exc_info=True)
                st.error(f"Error processing PDFs: {e}")
                processed_docs = []

        if processed_docs:
            with st.spinner("Indexing documents..."):
                try:
                    start_time = time.time()
                    st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                    logger.info(f"Vector store setup took {time.time() - start_time:.2f} seconds.")
                    st.session_state.processed_files = filenames
                    st.session_state.processed_documents = processed_docs
                    with st.spinner("Preparing extraction engines..."):
                        st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                        st.session_state.web_chain = create_web_extraction_chain(llm)
                        st.session_state.numind_chain = create_numind_extraction_chain()
                    logger.success("Extraction chains created.")
                    st.success(f"Successfully processed {len(filenames)} file(s).")
                except Exception as e:
                    logger.error(f"Vector store setup error: {e}", exc_info=True)
                    st.error(f"Error setting up vector store: {e}")
        else:
            st.warning("No text extracted from uploaded PDFs.")

elif process_button and not uploaded_files:
    st.warning("Please upload at least one PDF file.")

# Extraction logic
if st.session_state.pdf_chain and st.session_state.web_chain and not st.session_state.extraction_performed:
    if st.session_state.extraction_attempts > 3:
        logger.error("Too many extraction attempts. Resetting state.")
        st.error("Extraction loop detected. Please refresh and try again.")
        reset_evaluation_state()
        st.session_state.extraction_attempts = 0
        st.stop()

    st.session_state.extraction_attempts += 1
    logger.info(f"Starting extraction process (attempt {st.session_state.extraction_attempts})")
    part_number = st.session_state.get("part_number_input", "").strip()
    debug_logger.user_action("Extract button clicked", data={"attempt": st.session_state.extraction_attempts, "part_number": part_number}, context={"page": "extraction_attributs"})

    # Web scraping
    scraped_table_html = None
    if part_number:
        if st.session_state.current_part_number_scraped == part_number and st.session_state.scraped_table_html_cache:
            scraped_table_html = st.session_state.scraped_table_html_cache
            logger.info(f"Using cached scraped HTML for part number {part_number}.")
        else:
            with st.spinner("Scraping supplier websites..."):
                try:
                    scrape_start_time = time.time()
                    scraped_table_html = loop.run_until_complete(scrape_website_table_html(part_number))
                    scrape_time = time.time() - scrape_start_time
                    st.session_state.scraped_table_html_cache = scraped_table_html
                    st.session_state.current_part_number_scraped = part_number
                    if scraped_table_html:
                        logger.success(f"Web scraping successful in {scrape_time:.2f} seconds.")
                        st.caption(f"Found web data for part# {part_number}.")
                    else:
                        logger.warning(f"Web scraping failed in {scrape_time:.2f} seconds.")
                        st.caption(f"Web scraping failed for part# {part_number}.")
                except Exception as e:
                    logger.error(f"Web scraping error: {e}", exc_info=True)
                    st.warning(f"Web scraping error: {e}. Using PDF data only.")
                    st.session_state.scraped_table_html_cache = None
                    st.session_state.current_part_number_scraped = part_number
    else:
        logger.info("No part number provided, skipping web scrape.")
        st.session_state.scraped_table_html_cache = None
        st.session_state.current_part_number_scraped = None

    # Stage 1: Web Extraction
    intermediate_results = {}
    pdf_fallback_needed = []
    if scraped_table_html:
        cols = st.columns(2)
        col_index = 0
        for prompt_name, instructions in prompts_to_run.items():
            attribute_key = prompt_name
            web_instruction = instructions["web"]
            current_col = cols[col_index % 2]
            col_index += 1
            with current_col:
                with st.spinner(f"Stage 1: Extracting {attribute_key} from Web Data..."):
                    try:
                        start_time = time.time()
                        web_input = {
                            "cleaned_web_data": scraped_table_html,
                            "attribute_key": attribute_key,
                            "extraction_instructions": web_instruction
                        }
                        json_result_str = loop.run_until_complete(
                            _invoke_chain_and_process(st.session_state.web_chain, web_input, f"{attribute_key} (Web)")
                        )
                        run_time = time.time() - start_time
                        time.sleep(0.2)
                    except Exception as e:
                        logger.error(f"Stage 1 error for '{attribute_key}': {e}", exc_info=True)
                        json_result_str = f'{{"error": "Exception: {e}"}}'
                        run_time = time.time() - start_time

            final_answer_value = "Error"
            parse_error = None
            is_rate_limit = False
            raw_output = json_result_str or '{"error": "Stage 1 did not run"}'
            try:
                parsed_json = extract_json_from_string(raw_output.strip())
                if not isinstance(parsed_json, dict):
                    final_answer_value = "Unexpected JSON Type"
                    parse_error = TypeError(f"Expected dict, got {type(parsed_json)}")
                    pdf_fallback_needed.append(prompt_name)
                elif attribute_key in parsed_json:
                    parsed_value = str(parsed_json[attribute_key])
                    if "not found" in parsed_value.lower() or not parsed_value.strip():
                        final_answer_value = "NOT FOUND"
                        pdf_fallback_needed.append(prompt_name)
                    else:
                        final_answer_value = parsed_value
                elif "error" in parsed_json:
                    error_msg = parsed_json['error']
                    final_answer_value = f"Error: {error_msg[:100]}"
                    parse_error = ValueError(f"Stage 1 Error: {error_msg}")
                    pdf_fallback_needed.append(prompt_name)
                    if "rate limit" in error_msg.lower():
                        is_rate_limit = True
                else:
                    final_answer_value = "Unexpected JSON Format"
                    parse_error = ValueError(f"Unexpected JSON keys: {list(parsed_json.keys())}")
                    pdf_fallback_needed.append(prompt_name)
            except Exception as e:
                parse_error = e
                final_answer_value = "Processing Error"
                pdf_fallback_needed.append(prompt_name)

            intermediate_results[prompt_name] = {
                'Prompt Name': prompt_name,
                'Extracted Value': final_answer_value,
                'Ground Truth': '',
                'Source': 'Web',
                'Raw Output': raw_output,
                'Parse Error': str(parse_error) if parse_error else None,
                'Is Success': not parse_error and final_answer_value != "NOT FOUND" and not is_rate_limit,
                'Is Error': bool(parse_error) and not is_rate_limit,
                'Is Not Found': final_answer_value == "NOT FOUND",
                'Is Rate Limit': is_rate_limit,
                'Latency (s)': round(run_time, 2),
                'Exact Match': None,
                'Case-Insensitive Match': None
            }
    else:
        pdf_fallback_needed = list(prompts_to_run.keys())
        for prompt_name in pdf_fallback_needed:
            intermediate_results[prompt_name] = {
                'Prompt Name': prompt_name,
                'Extracted Value': "(Web Stage Skipped)",
                'Ground Truth': '',
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

    # Stage 2: NuMind Fallback
    st.info(f"Running Stage 2 (NuMind Fallback) for {len(pdf_fallback_needed)} attributes...")
    if pdf_fallback_needed:
        if not st.session_state.numind_chain:
            st.warning("NuMind extraction unavailable. Falling back to PDF extraction.")
            cols = st.columns(2)
            col_index = 0
            for prompt_name in pdf_fallback_needed:
                attribute_key = prompt_name
                pdf_instruction = prompts_to_run[attribute_key]["pdf"]
                current_col = cols[col_index % 2]
                col_index += 1
                with current_col:
                    with st.spinner(f"Stage 2: Extracting {attribute_key} from PDF Data..."):
                        try:
                            start_time = time.time()
                            context_chunks = fetch_chunks(st.session_state.retriever, part_number, attribute_key, k=8)
                            context_text = "\n\n".join([chunk.page_content for chunk in context_chunks]) if context_chunks else ""
                            pdf_input = {
                                "context": context_text,
                                "extraction_instructions": pdf_instruction,
                                "attribute_key": attribute_key,
                                "part_number": part_number or "Not Provided"
                            }
                            json_result_str = loop.run_until_complete(
                                _invoke_chain_and_process(st.session_state.pdf_chain, pdf_input, f"{attribute_key} (PDF)")
                            )
                            run_time = time.time() - start_time
                            time.sleep(0.5)
                        except Exception as e:
                            logger.error(f"Stage 2 error for '{attribute_key}': {e}", exc_info=True)
                            json_result_str = f'{{"error": "Exception: {e}"}}'
                            run_time = time.time() - start_time

                final_answer_value = "Error"
                parse_error = None
                raw_output = json_result_str or '{"error": "Stage 2 did not run"}'
                try:
                    parsed_json = extract_json_from_string(raw_output.strip())
                    if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                        final_answer_value = str(parsed_json[attribute_key])
                    elif isinstance(parsed_json, dict) and "error" in parsed_json:
                        final_answer_value = f"Error: {parsed_json['error'][:100]}"
                        parse_error = ValueError(f"Stage 2 Error: {parsed_json['error']}")
                    else:
                        final_answer_value = "Unexpected JSON Format"
                        parse_error = ValueError("Stage 2 Unexpected JSON format")
                except Exception as e:
                    parse_error = e
                    final_answer_value = "Processing Error"

                intermediate_results[prompt_name].update({
                    'Extracted Value': final_answer_value,
                    'Source': 'PDF',
                    'Raw Output': raw_output,
                    'Parse Error': str(parse_error) if parse_error else None,
                    'Is Success': not parse_error and "not found" not in final_answer_value.lower() and final_answer_value.strip(),
                    'Is Error': bool(parse_error),
                    'Is Not Found': "not found" in final_answer_value.lower() or not final_answer_value.strip(),
                    'Is Rate Limit': False,
                    'Latency (s)': intermediate_results[prompt_name].get('Latency (s)', 0.0) + round(run_time, 2)
                })
        else:
            st.success("Using NuMind for structured extraction...")
            file_data = st.session_state.uploaded_file_data[0][1] if st.session_state.uploaded_file_data else None
            if not file_data:
                st.error("No file data for NuMind extraction.")
            else:
                with st.spinner("Running NuMind extraction..."):
                    try:
                        start_time = time.time()
                        extraction_schema = get_custom_schema()
                        numind_result = loop.run_until_complete(
                            extract_with_numind_using_schema(st.session_state.numind_chain, file_data, extraction_schema)
                        )
                        run_time = time.time() - start_time
                        if numind_result:
                            st.success(f"NuMind extraction completed in {run_time:.2f} seconds.")
                            for prompt_name in pdf_fallback_needed:
                                attribute_key = prompt_name
                                final_answer_value = extract_specific_attribute_from_numind_result(numind_result, attribute_key) or "NOT FOUND"
                                intermediate_results[prompt_name].update({
                                    'Extracted Value': final_answer_value,
                                    'Source': 'NuMind',
                                    'Raw Output': json.dumps(numind_result) if numind_result else "No NuMind result",
                                    'Parse Error': None,
                                    'Is Success': final_answer_value != "NOT FOUND",
                                    'Is Error': False,
                                    'Is Not Found': final_answer_value == "NOT FOUND",
                                    'Is Rate Limit': False,
                                    'Latency (s)': intermediate_results[prompt_name].get('Latency (s)', 0.0) + round(run_time, 2)
                                })
                        else:
                            st.error("NuMind extraction failed.")
                            for prompt_name in pdf_fallback_needed:
                                intermediate_results[prompt_name].update({
                                    'Extracted Value': "NuMind Extraction Failed",
                                    'Source': 'NuMind',
                                    'Raw Output': "NuMind API returned no results",
                                    'Parse Error': "NuMind extraction failed",
                                    'Is Success': False,
                                    'Is Error': True,
                                    'Is Not Found': False,
                                    'Is Rate Limit': False,
                                    'Latency (s)': round(run_time, 2)
                                })
                    except Exception as e:
                        st.error(f"NuMind extraction error: {e}")
                        for prompt_name in pdf_fallback_needed:
                            intermediate_results[prompt_name].update({
                                'Extracted Value': f"NuMind Error: {str(e)[:100]}",
                                'Source': 'NuMind',
                                'Raw Output': f"Exception: {e}",
                                'Parse Error': str(e),
                                'Is Success': False,
                                'Is Error': True,
                                'Is Not Found': False,
                                'Is Rate Limit': False,
                                'Latency (s)': round(time.time() - start_time, 2)
                            })

    # Stage 3: Final Fallback
    extraction_results_list = list(intermediate_results.values())
    final_fallback_needed = [
        result['Prompt Name'] for result in extraction_results_list
        if result.get('Is Not Found', False) or result.get('Extracted Value', '') in ["NOT FOUND", "Error", "Processing Error", "Unexpected JSON Format", "Unexpected JSON Type", "(Web Stage Skipped)"] or
           not result.get('Extracted Value', '').strip() or result.get('Extracted Value', '').lower() in ["none", "null", "n/a", "na"]
    ]
    if final_fallback_needed:
        none_responses = [r['Prompt Name'] for r in extraction_results_list if r.get('Extracted Value', '').lower() in ["none", "null", "n/a", "na"]]
        render_extraction_progress(len(prompts_to_run), len(pdf_fallback_needed), len(final_fallback_needed), None, none_responses)
        cols = st.columns(2)
        col_index = 0
        for prompt_name in final_fallback_needed:
            attribute_key = prompt_name
            pdf_instruction = prompts_to_run[attribute_key]["pdf"]
            current_col = cols[col_index % 2]
            col_index += 1
            with current_col:
                with st.spinner(f"Stage 3: Final recheck for {attribute_key}..."):
                    try:
                        start_time = time.time()
                        context_chunks = fetch_chunks(st.session_state.retriever, part_number, attribute_key, k=12)
                        context_text = "\n\n".join([chunk.page_content for chunk in context_chunks]) if context_chunks else ""
                        previous_value = next((r['Extracted Value'] for r in extraction_results_list if r['Prompt Name'] == prompt_name), '')
                        enhanced_instruction = f"{pdf_instruction}\n\nCRITICAL: Previous extraction returned '{previous_value}'. Be thorough and look for any mention of this attribute." if previous_value.lower() in ["none", "null", "n/a", "na"] else f"{pdf_instruction}\n\nIMPORTANT: Final recheck. Be thorough."
                        pdf_input = {
                            "context": context_text,
                            "extraction_instructions": enhanced_instruction,
                            "attribute_key": attribute_key,
                            "part_number": part_number or "Not Provided"
                        }
                        json_result_str = loop.run_until_complete(
                            _invoke_chain_and_process(st.session_state.pdf_chain, pdf_input, f"{attribute_key} (Final Fallback)")
                        )
                        run_time = time.time() - start_time
                        time.sleep(0.3)
                    except Exception as e:
                        logger.error(f"Stage 3 error for '{attribute_key}': {e}", exc_info=True)
                        json_result_str = f'{{"error": "Exception: {e}"}}'
                        run_time = time.time() - start_time

            final_answer_value = "Error"
            parse_error = None
            raw_output = json_result_str or '{"error": "Stage 3 did not run"}'
            try:
                parsed_json = extract_json_from_string(raw_output.strip())
                if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                    parsed_value = str(parsed_json[attribute_key])
                    final_answer_value = "NOT FOUND (Final)" if not parsed_value.strip() or "not found" in parsed_value.lower() or parsed_value.lower() in ["none", "null", "n/a", "na"] else parsed_value
                elif isinstance(parsed_json, dict) and "error" in parsed_json:
                    final_answer_value = f"Error: {parsed_json['error'][:100]}"
                    parse_error = ValueError(f"Stage 3 Error: {parsed_json['error']}")
                else:
                    final_answer_value = "Unexpected JSON Format (Final)"
                    parse_error = ValueError("Stage 3 Unexpected JSON format")
            except Exception as e:
                parse_error = e
                final_answer_value = "Processing Error (Final)"

            for i, result in enumerate(extraction_results_list):
                if result['Prompt Name'] == prompt_name:
                    original_value = result['Extracted Value']
                    original_source = result['Source']
                    should_rollback = (
                        (original_value.lower() in ["none", "null", "n/a", "na"] and final_answer_value == "NOT FOUND (Final)") or
                        bool(parse_error) or
                        final_answer_value in ["Error", "Processing Error (Final)", "Unexpected JSON Format (Final)"]
                    )
                    final_display_value = original_value if should_rollback else final_answer_value
                    final_source = original_source if should_rollback else 'Final Fallback'
                    extraction_results_list[i].update({
                        'Extracted Value': final_display_value,
                        'Source': final_source,
                        'Raw Output': raw_output if not should_rollback else result['Raw Output'],
                        'Parse Error': str(parse_error) if parse_error and not should_rollback else result['Parse Error'],
                        'Is Success': not bool(parse_error) and final_answer_value not in ["NOT FOUND (Final)", "Error", "Processing Error (Final)", "Unexpected JSON Format (Final)"],
                        'Is Error': bool(parse_error),
                        'Is Not Found': final_answer_value in ["NOT FOUND (Final)"],
                        'Is Rate Limit': False,
                        'Latency (s)': result.get('Latency (s)', 0.0) + round(run_time, 2)
                    })
                    break

    # Stage Summary
    st.divider()
    st.subheader("📊 Extraction Stage Summary")
    stage_summary = {}
    for result in extraction_results_list:
        source = result.get('Source', 'Unknown')
        stage_summary.setdefault(source, {'total': 0, 'success': 0, 'error': 0, 'not_found': 0})
        stage_summary[source]['total'] += 1
        if result.get('Is Success', False):
            stage_summary[source]['success'] += 1
        elif result.get('Is Error', False):
            stage_summary[source]['error'] += 1
        elif result.get('Is Not Found', False):
            stage_summary[source]['not_found'] += 1

    summary_cols = st.columns(len(stage_summary))
    for i, (source, stats) in enumerate(stage_summary.items()):
        with summary_cols[i]:
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            color = "#28a745" if success_rate > 70 else "#ffc107" if success_rate > 30 else "#dc3545"
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}80 100%); 
                            color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4 style="margin: 0;">{source}</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9em;">
                        Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)<br>
                        Errors: {stats['error']} | Not Found: {stats['not_found']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.session_state.evaluation_results = extraction_results_list
    st.session_state.extraction_performed = True
    st.session_state.extraction_attempts = 0
    st.success("Extraction complete. Enter ground truth below.")

# Display Results
if st.session_state.evaluation_results:
    extracted_data = {result['Prompt Name']: result['Extracted Value'] for result in st.session_state.evaluation_results
                     if result.get('Extracted Value') and result['Extracted Value'] not in ['NOT FOUND', 'ERROR']}
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1rem; border-radius: 15px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.5em;">📊 Extraction Results</h3>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="extraction-results">', unsafe_allow_html=True)
    for key, value in extracted_data.items():
        display_value = value[:100] + "..." if len(value) > 100 else value
        st.markdown(f"""
            <div class="result-item">
                <div class="result-label">🔍 {key}</div>
                <div class="result-value" title="{value}">{display_value}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.evaluation_metrics:
        metrics = st.session_state.evaluation_metrics
        st.markdown("""
            <div style="background: white; border-radius: 15px; padding: 1rem; box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);">
                <h4 style="color: #1e3c72;">📈 Success Metrics</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        with col2:
            st.metric("Total Fields", metrics.get('total_fields', 0))
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Upload and process documents to see extracted results.")

# Manual Recheck
st.divider()
st.subheader("🔄 Manual Attribute Recheck")
with st.expander("ℹ️ How to use Manual Recheck"):
    st.markdown("""
        **Manual Recheck allows you to re-extract specific attributes:**
        1. Select attributes from the dropdown.
        2. Click 'Run Manual Recheck'.
        3. Review updated results.
        **When to use:** For attributes with 'NOT FOUND', 'none', or errors.
    """)

manual_recheck_candidates = [
    result['Prompt Name'] for result in st.session_state.evaluation_results
    if result.get('Is Not Found', False) or result.get('Extracted Value', '') in ["NOT FOUND", "NOT FOUND (Final)", "Error", "Processing Error", "Processing Error (Final)", "Unexpected JSON Format", "Unexpected JSON Format (Final)", "Unexpected JSON Type", "(Web Stage Skipped)"] or
       not result.get('Extracted Value', '').strip() or result.get('Extracted Value', '').lower() in ["none", "null", "n/a", "na"]
]
if manual_recheck_candidates:
    st.info(f"Found {len(manual_recheck_candidates)} attributes for manual recheck.")
    selected_for_recheck = st.multiselect(
        "Select attributes to recheck:",
        options=manual_recheck_candidates,
        default=manual_recheck_candidates[:3]
    )
    if selected_for_recheck and st.button("🔄 Run Manual Recheck", type="primary"):
        for prompt_name in selected_for_recheck:
            attribute_key = prompt_name
            pdf_instruction = prompts_to_run[attribute_key]["pdf"]
            with st.spinner(f"Manual recheck for {attribute_key}..."):
                try:
                    start_time = time.time()
                    context_chunks = fetch_chunks(st.session_state.retriever, part_number, attribute_key, k=15)
                    context_text = "\n\n".join([chunk.page_content for chunk in context_chunks]) if context_chunks else ""
                    previous_value = next((r['Extracted Value'] for r in st.session_state.evaluation_results if r['Prompt Name'] == prompt_name), '')
                    manual_instruction = f"{pdf_instruction}\n\nMANUAL RECHECK: Previous result '{previous_value}'. Be exhaustive." if previous_value.lower() in ["none", "null", "n/a", "na"] else f"{pdf_instruction}\n\nMANUAL RECHECK: Be thorough."
                    manual_recheck_input = {
                        "context": context_text,
                        "extraction_instructions": manual_instruction,
                        "attribute_key": attribute_key,
                        "part_number": part_number or "Not Provided"
                    }
                    json_result_str = loop.run_until_complete(
                        _invoke_chain_and_process(st.session_state.pdf_chain, manual_recheck_input, f"{attribute_key} (Manual Recheck)")
                    )
                    run_time = time.time() - start_time
                    final_answer_value = "Error"
                    parse_error = None
                    raw_output = json_result_str or '{"error": "Manual recheck did not run"}'
                    try:
                        parsed_json = extract_json_from_string(raw_output.strip())
                        if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                            parsed_value = str(parsed_json[attribute_key])
                            final_answer_value = "NOT FOUND (Manual)" if not parsed_value.strip() or "not found" in parsed_value.lower() or parsed_value.lower() in ["none", "null", "n/a", "na"] else parsed_value
                        elif isinstance(parsed_json, dict) and "error" in parsed_json:
                            final_answer_value = f"Error: {parsed_json['error'][:100]}"
                            parse_error = ValueError(f"Manual Recheck Error: {parsed_json['error']}")
                        else:
                            final_answer_value = "Unexpected JSON Format (Manual)"
                            parse_error = ValueError("Manual Recheck Unexpected JSON format")
                    except Exception as e:
                        parse_error = e
                        final_answer_value = "Processing Error (Manual)"
                    for i, result in enumerate(st.session_state.evaluation_results):
                        if result['Prompt Name'] == prompt_name:
                            original_value = result['Extracted Value']
                            original_source = result['Source']
                            should_rollback = (
                                (original_value.lower() in ["none", "null", "n/a", "na"] and final_answer_value == "NOT FOUND (Manual)") or
                                bool(parse_error) or
                                final_answer_value in ["Error", "Processing Error (Manual)", "Unexpected JSON Format (Manual)"]
                            )
                            final_display_value = original_value if should_rollback else final_answer_value
                            final_source = original_source if should_rollback else 'Manual Recheck'
                            st.session_state.evaluation_results[i].update({
                                'Extracted Value': final_display_value,
                                'Source': final_source,
                                'Raw Output': raw_output if not should_rollback else result['Raw Output'],
                                'Parse Error': str(parse_error) if parse_error and not should_rollback else result['Parse Error'],
                                'Is Success': not bool(parse_error) and final_answer_value not in ["NOT FOUND (Manual)", "Error", "Processing Error (Manual)", "Unexpected JSON Format (Manual)"],
                                'Is Error': bool(parse_error),
                                'Is Not Found': final_answer_value in ["NOT FOUND (Manual)"],
                                'Is Rate Limit': False,
                                'Latency (s)': result.get('Latency (s)', 0.0) + round(run_time, 2)
                            })
                            if should_rollback and bool(parse_error):
                                st.warning(f"Rolled back to original '{original_value}' for '{attribute_key}' (manual recheck error)")
                            elif should_rollback and original_value.lower() in ["none", "null", "n/a", "na"]:
                                st.info(f"Preserved original '{original_value}' for '{attribute_key}' (confirmed by manual recheck)")
                            break
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Manual recheck error for '{attribute_key}': {e}")
        st.success("Manual recheck completed!")
        st.rerun()
else:
    st.success("All attributes extracted successfully! No manual recheck needed.")

# Raw Extraction View
if st.session_state.processed_documents:
    st.divider()
    st.header("4. View Raw Mistral Extraction")
    if st.button("👁️ View Raw Extracted Document Content", key="view_raw_extraction"):
        st.session_state.show_raw_extraction = not st.session_state.get('show_raw_extraction', False)
    if st.session_state.get('show_raw_extraction', False):
        st.info("Raw document content extracted by Mistral Vision.")
        docs_by_source = {}
        for doc in st.session_state.processed_documents:
            source = doc.metadata.get('source', 'Unknown')
            docs_by_source.setdefault(source, []).append(doc)
        for source, docs in docs_by_source.items():
            st.subheader(f"📄 {source}")
            docs.sort(key=lambda x: x.metadata.get('page', 0))
            for doc in docs:
                page_num = doc.metadata.get('page', 'Unknown')
                st.markdown(f"**Page {page_num}:**")
                st.code(doc.page_content, language='markdown')
                with st.expander(f"Page {page_num} Metadata"):
                    st.json(doc.metadata)
                st.divider()
            if st.button("📥 Download Raw Extraction (Markdown)", key="download_raw_extraction"):
                raw_content = f"# Raw Mistral Extraction Report\n\n**Extraction Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n**Total Documents:** {len(st.session_state.processed_documents)}\n\n"
                for source, docs in docs_by_source.items():
                    raw_content += f"## 📄 {source}\n\n"
                    docs.sort(key=lambda x: x.metadata.get('page', 0))
                    for doc in docs:
                        page_num = doc.metadata.get('page', 'Unknown')
                        raw_content += f"### Page {page_num}\n\n**Metadata:**\n```json\n{json.dumps(doc.metadata, indent=2)}\n```\n\n**Content:**\n```markdown\n{doc.page_content}\n```\n\n---\n\n"
                st.download_button(
                    label="📄 Download Raw Extraction Report",
                    data=raw_content.encode('utf-8'),
                    file_name='raw_mistral_extraction.md',
                    mime='text/markdown',
                    key='download_raw_md'
                )

# Export Results
st.divider()
st.header("6. Export Results")
if st.session_state.evaluation_results:
    export_df = pd.DataFrame(st.session_state.evaluation_results)
    export_summary = st.session_state.evaluation_metrics or {}
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
    st.info("Process documents to enable export.")