# --- Force python to use pysqlite3 based on chromadb docs ---
# This override MUST happen before any other imports that might import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# app.py
# Remove the page config since it's handled by main.py
# st.set_page_config(
#     page_title="Connector Data Extraction",
#     page_icon="🔌",
#     layout="wide",
#     initial_sidebar_state="expanded"v
# )

import os
import time
from loguru import logger
import json # Import the json library
import pandas as pd # Add pandas import
import re # Import the 're' module for regular expressions
import asyncio # Add asyncio import
import subprocess # To run playwright install
import nest_asyncio # Add nest_asyncio for better async handling
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# --- Install Playwright browsers needed by crawl4ai --- 
# This should run on startup in the Streamlit Cloud environment
def install_playwright_browsers():
    logger.info("Checking and installing Playwright browsers if needed...")
    try:
        # Use subprocess to run the command
        # stdout/stderr=subprocess.PIPE can capture output if needed
        # check=True will raise an error if the command fails
        process = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                                 capture_output=True, text=True, check=False) # Use check=False initially to see output
        if process.returncode == 0:
             logger.success("Playwright browsers installed successfully (or already exist).")
        else:
             # Log stdout/stderr for debugging if it failed
             logger.error(f"Playwright browser install command failed with code {process.returncode}.")
             logger.error(f"stdout: {process.stdout}")
             logger.error(f"stderr: {process.stderr}")
             # Optionally raise an error or show a Streamlit warning
             # st.warning("Failed to install necessary Playwright browsers. Web scraping may fail.")
        # Alternative using playwright's internal API (might be cleaner if stable)
        # from playwright.driver import main as playwright_main
        # playwright_main.main(['install']) # Installs default browser (chromium)
        # logger.success("Playwright browsers installed successfully via internal API.")
    except FileNotFoundError:
        logger.error("Could not find 'playwright' command. Is playwright installed correctly?")
        st.error("Playwright not found. Please ensure 'playwright' is in requirements.txt")
    except Exception as e:
        logger.error(f"An error occurred during Playwright browser installation: {e}", exc_info=True)
        st.warning(f"An error occurred installing Playwright browsers: {e}. Web scraping may fail.")

# Only install if not already installed
if 'playwright_installed' not in st.session_state:
    install_playwright_browsers()
    st.session_state.playwright_installed = True

# Import project modules
import config
from pdf_processor import process_uploaded_pdfs
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
# Updated imports from llm_interface
from llm_interface import (
    initialize_llm,
    create_pdf_extraction_chain, # Use PDF chain func
    create_web_extraction_chain, # Use Web chain func
    _invoke_chain_and_process, # Use the helper directly
    scrape_website_table_html
)
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
# Import the NEW web prompts
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

# --- Cached Resource Functions ---
@st.cache_resource
def initialize_embeddings():
    """Initialize and cache the embedding function"""
    try:
        embedding_function = get_embedding_function()
        return embedding_function
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        return None

@st.cache_resource
def initialize_llm_cached():
    """Initialize and cache the LLM"""
    try:
        llm = initialize_llm()
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        return None

def main():
    # --- Nouvelle page d'accueil stylisée ---
    st.set_page_config(page_title="LEOPARTS", page_icon="🦁", layout="wide")

    # Masquer le menu automatique Streamlit dans la sidebar
    st.markdown(
        """<style>
        [data-testid="stSidebarNav"] {display: none;}
        </style>""",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
        if st.button("🏠 Home"):
            st.switch_page("app.py")
        if st.button("🤖 Chat with Leoparts"):
            st.switch_page("pages/chatbot.py")
        if st.button("📄 Extract a new Part"):
            st.switch_page("pages/extraction_attributs.py")

    # Main welcome content
    st.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh;'>
            <h1 style='font-size: 3em; margin-bottom: 0.2em;'>LEOPARTS</h1>
            <h2 style='font-size: 2em; margin-bottom: 0.5em;'>Welcome!</h2>
            <p style='font-size: 1.5em; margin-bottom: 2em;'>Choose a Tool</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 Chat with Leoparts", key="main_chat_btn", use_container_width=True):
                st.switch_page("pages/chatbot.py")
        with c2:
            if st.button("📄 Extract a new Part", key="main_extract_btn", use_container_width=True):
                st.switch_page("pages/extraction_attributs.py")

    # Initialize session state
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'pdf_chain' not in st.session_state:
        st.session_state.pdf_chain = None
    if 'web_chain' not in st.session_state:
        st.session_state.web_chain = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'extraction_performed' not in st.session_state:
        st.session_state.extraction_performed = False
    if 'scraped_table_html_cache' not in st.session_state:
        st.session_state.scraped_table_html_cache = None
    if 'current_part_number_scraped' not in st.session_state:
        st.session_state.current_part_number_scraped = None

    # Initialize embeddings
    try:
        logger.info("Attempting to initialize embedding function...")
        embedding_function = initialize_embeddings()
        if embedding_function:
             logger.success("Embedding function initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize embedding model. Error: {e}")
        st.stop()

    # Initialize LLM
    try:
        logger.info("Attempting to initialize LLM...")
        llm = initialize_llm_cached()
        if llm:
            logger.success("LLM initialized successfully.")
    except Exception as e:
         logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
         st.error(f"Fatal Error: Could not initialize LLM. Error: {e}")
         st.stop()

    # Check if initializations failed
    if embedding_function is None or llm is None:
         if not st.exception: # If st.stop() wasn't called already
            st.error("Core components (Embeddings or LLM) failed to initialize. Cannot continue.")
         st.stop()

    # Load existing data if available
    if st.session_state.retriever is None and config.CHROMA_SETTINGS.is_persistent and embedding_function:
        logger.info("Attempting to load existing vector store...")
        st.session_state.retriever = load_existing_vector_store(embedding_function)
        if st.session_state.retriever:
            logger.success("Successfully loaded retriever from persistent storage.")
            st.session_state.processed_files = ["Existing data loaded from disk"]
            # --- Create BOTH Extraction Chains --- 
            logger.info("Creating extraction chains from loaded retriever...")
            st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
            st.session_state.web_chain = create_web_extraction_chain(llm)
            if not st.session_state.pdf_chain or not st.session_state.web_chain:
                st.warning("Failed to create one or both extraction chains from loaded retriever.")
            # ------------------------------------
            # Don't reset evaluation if loading existing data, but ensure extraction hasn't run yet
            st.session_state.extraction_performed = False # Ensure flag is false on load
        else:
            logger.warning("No existing persistent vector store found or failed to load.")




    # ... rest of the extraction results rendering code ...

if __name__ == "__main__":
    main()