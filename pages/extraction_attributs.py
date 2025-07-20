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
    """
    <style>
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
    .stepper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0 2rem 0;
    }
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 2rem;
        position: relative;
    }
    .step-icon {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #e3eafc;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2em;
        margin-bottom: 0.5rem;
        animation: pulse 1.2s infinite;
    }
    .step-icon.done {
        background: #28a745;
        color: white;
        animation: none;
    }
    .step-icon.active {
        background: #007bff;
        color: white;
        animation: bounce 0.8s infinite alternate;
    }
    .step-label {
        font-size: 1em;
        color: #1e3c72;
        font-weight: 600;
        text-align: center;
    }
    .step-connector {
        position: absolute;
        top: 24px;
        left: 100%;
        width: 60px;
        height: 4px;
        background: #b0c4de;
        z-index: 0;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 #4a90e2; }
        70% { box-shadow: 0 0 0 10px rgba(74,144,226,0); }
        100% { box-shadow: 0 0 0 0 rgba(74,144,226,0); }
    }
    @keyframes bounce {
        0% { transform: translateY(0); }
        100% { transform: translateY(-8px); }
    }
    .card-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin: 2rem 0;
        justify-content: center;
    }
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #1e3c72;
        border-radius: 16px;
        padding: 1.5rem;
        min-width: 280px;
        max-width: 340px;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.08);
        transition: box-shadow 0.2s, transform 0.2s;
        cursor: pointer;
        position: relative;
    }
    .result-card:hover {
        box-shadow: 0 8px 30px rgba(30, 60, 114, 0.18);
        transform: translateY(-4px) scale(1.03);
    }
    .card-title {
        color: #1e3c72;
        font-size: 1.2em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 1.1em;
        color: #2a5298;
        margin-bottom: 0.5rem;
        word-break: break-all;
    }
    .card-details {
        background: #fff;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(30, 60, 114, 0.06);
        font-size: 0.98em;
        color: #333;
    }
    .hide-details { display: none; }
    </style>
    """,
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

# --- Stepper Example (replace old stepper/summary UI) ---
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = 'upload'
if 'stepper_stage' not in st.session_state:
    st.session_state.stepper_stage = 0

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

if st.session_state.processing_state == 'processing':
    st.markdown("""
        <div class="stepper">
            <div class="step">
                <div class="step-icon done">1</div>
                <div class="step-label">Upload</div>
                <div class="step-connector"></div>
            </div>
            <div class="step">
                <div class="step-icon active">2</div>
                <div class="step-label">Processing</div>
                <div class="step-connector"></div>
            </div>
            <div class="step">
                <div class="step-icon">3</div>
                <div class="step-label">Results</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    # ... existing processing logic ...

if st.session_state.processing_state == 'results':
    st.markdown("""
        <div class="stepper">
            <div class="step">
                <div class="step-icon done">1</div>
                <div class="step-label">Upload</div>
                <div class="step-connector"></div>
            </div>
            <div class="step">
                <div class="step-icon done">2</div>
                <div class="step-label">Processing</div>
                <div class="step-connector"></div>
            </div>
            <div class="step">
                <div class="step-icon active">3</div>
                <div class="step-label">Results</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.success("All done! Here are the details we found in your document. Click any card to see more.")
    results = st.session_state.get('evaluation_results', [])
    if results:
        st.markdown('<div class="card-grid">', unsafe_allow_html=True)
        for i, result in enumerate(results):
            if isinstance(result, dict):
                prompt_name = result.get('Prompt Name', 'Unknown')
                extracted_value = result.get('Extracted Value', '')
                st.markdown(f'''
                <div class="result-card">
                    <div class="card-title">{prompt_name}</div>
                    <div class="card-value">{extracted_value}</div>
                    <details class="card-details">
                        <summary>Show details</summary>
                        <b>Source:</b> {result.get('Source', 'N/A')}<br>
                        <b>Processing time:</b> {result.get('Latency (s)', 'N/A')} seconds<br>
                        <b>Raw Output:</b><br>
                        <pre style='font-size:0.95em; background:#f4f4f4; border-radius:6px; padding:0.5em;'>{result.get('Raw Output', '')[:500]}</pre>
                    </details>
                </div>
                ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No results found. Try uploading a different document.")
    if st.button("Start Over"):
        st.session_state.processing_state = 'upload'
        st.session_state.stepper_stage = 0
        st.session_state.expanded_card = None
        st.rerun()

    # --- Block 2: Display Ground Truth / Metrics (if results exist) ---
    # This part needs the 'Source' column re-added for display
    if st.session_state.evaluation_results:
        debug_logger.info("Displaying evaluation results", data={
            "results_count": len(st.session_state.evaluation_results)
        }, context={"step": "display_results"})
        
        st.divider()
        st.header("3. Enter Ground Truth & Evaluate")

        results_df = pd.DataFrame(st.session_state.evaluation_results)
        
        debug_logger.data_transformation(
            "Results DataFrame creation",
            st.session_state.evaluation_results,
            results_df.to_dict('records'),
            context={"step": "results_dataframe_created"}
        )
        
        if 'Source' not in results_df.columns:
             results_df['Source'] = 'Unknown' # Add placeholder if missing
             debug_logger.warning("Source column missing, added placeholder", context={"step": "source_column_fixed"})

        st.info("Enter the correct 'Ground Truth' value for each field below. Leave blank if the field shouldn't exist or 'NOT FOUND' is correct.")

        disabled_cols = [col for col in results_df.columns if col != 'Ground Truth']
        column_order = [ # Add Source back
            'Prompt Name', 'Extracted Value', 'Ground Truth', 'Source',
            'Is Success', 'Is Error', 'Is Not Found', 'Is Rate Limit',
            'Latency (s)', 'Exact Match', 'Case-Insensitive Match'
        ]

        edited_df = st.data_editor(
            results_df,
            key="gt_editor",
            use_container_width=True,
            num_rows="dynamic",
            disabled=disabled_cols,
            column_order=column_order,
            column_config={ # Add Source back
                 "Prompt Name": st.column_config.TextColumn(width="medium"),
                 "Extracted Value": st.column_config.TextColumn(width="medium"),
                 "Ground Truth": st.column_config.TextColumn(width="medium", help="Enter the correct value here"),
                 "Source": st.column_config.TextColumn(width="small"), # Show source
                 "Is Success": st.column_config.CheckboxColumn("Success?", width="small"),
                 "Is Error": st.column_config.CheckboxColumn("Error?", width="small"),
                 "Is Not Found": st.column_config.CheckboxColumn("Not Found?", width="small"),
                 "Is Rate Limit": st.column_config.CheckboxColumn("Rate Limit?", width="small"),
                 "Latency (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                 "Exact Match": st.column_config.CheckboxColumn("Exact?", width="small"),
                 "Case-Insensitive Match": st.column_config.CheckboxColumn("Case-Ins?", width="small"),
                 "Raw Output": None,
                 "Parse Error": None
            }
        )
        
        debug_logger.user_action("Data editor displayed", data={
            "df_shape": edited_df.shape,
            "columns": list(edited_df.columns)
        }, context={"step": "data_editor_displayed"})

        # --- Mini Debug Widget ---
        from debug_interface import create_mini_debug_widget
        create_mini_debug_widget()
        
        # --- Manual Recheck Section ---
        st.divider()
        st.subheader("🔄 Manual Attribute Recheck")
        
        # Help section
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
        
        # Get attributes that might need manual recheck
        manual_recheck_candidates = []
        for result in st.session_state.evaluation_results:
            if isinstance(result, dict):
                extracted_value = result.get('Extracted Value', '')
                is_not_found = result.get('Is Not Found', False)
                is_error = result.get('Is Error', False)
                
                if (is_not_found or 
                    extracted_value in ["NOT FOUND", "NOT FOUND (Final)", "Error", "Processing Error", "Processing Error (Final)", "Unexpected JSON Format", "Unexpected JSON Format (Final)", "Unexpected JSON Type"] or
                    not extracted_value or 
                    extracted_value.strip() == "" or
                    extracted_value == "(Web Stage Skipped)" or
                    extracted_value.lower() in ["none", "null", "n/a", "na"]):  # Also include "none" responses for manual recheck
                    manual_recheck_candidates.append(result.get('Prompt Name', ''))
        
        if manual_recheck_candidates:
            # Count "none" responses in manual recheck candidates
            none_candidates = []
            other_candidates = []
            for result in st.session_state.evaluation_results:
                if isinstance(result, dict) and result.get('Prompt Name') in manual_recheck_candidates:
                    extracted_value = result.get('Extracted Value', '')
                    if extracted_value.lower() in ["none", "null", "n/a", "na"]:
                        none_candidates.append(result.get('Prompt Name'))
                    else:
                        other_candidates.append(result.get('Prompt Name'))
            
            st.info(f"Found {len(manual_recheck_candidates)} attributes that might benefit from manual recheck.")
            if none_candidates:
                st.warning(f"⚠️ {len(none_candidates)} of these returned 'none' responses and may contain missed values.")
            
            # Allow user to select specific attributes for recheck
            selected_for_recheck = st.multiselect(
                "Select attributes to recheck:",
                options=manual_recheck_candidates,
                default=manual_recheck_candidates[:3],  # Default to first 3
                help="Select attributes that returned 'NOT FOUND' or errors for additional extraction attempts."
            )
            
            if selected_for_recheck and st.button("🔄 Run Manual Recheck", type="primary"):
                st.info(f"Running manual recheck for {len(selected_for_recheck)} selected attributes...")
                
                # Run manual recheck
                for prompt_name in selected_for_recheck:
                    attribute_key = prompt_name
                    pdf_instruction = prompts_to_run[attribute_key]["pdf"]
                    
                    with st.spinner(f"Manual recheck for {attribute_key}..."):
                        try:
                            start_time = time.time()
                            
                            # Use even more chunks for manual recheck
                            context_chunks = fetch_chunks(
                                st.session_state.retriever,
                                part_number,
                                attribute_key,
                                k=15  # Increased for thorough manual recheck
                            )
                            context_text = "\n\n".join([chunk.page_content for chunk in context_chunks]) if context_chunks else ""
                            
                            # Enhanced prompt for manual recheck
                            # Check if this attribute previously returned "none" or similar
                            previous_value = None
                            for result in st.session_state.evaluation_results:
                                if result.get('Prompt Name') == prompt_name:
                                    previous_value = result.get('Extracted Value', '')
                                    break
                            
                            # Customize manual recheck prompt based on previous result
                            if previous_value and previous_value.lower() in ["none", "null", "n/a", "na"]:
                                manual_instruction = f"{pdf_instruction}\n\nMANUAL RECHECK - CRITICAL: Previous extraction returned '{previous_value}'. This may be incorrect. Please be extremely thorough and look for ANY mention of this attribute, even if it's not explicitly labeled. Consider technical specifications, material properties, dimensions, or any related information that might indicate this attribute's value. This is a manual recheck request - be exhaustive in your search."
                            else:
                                manual_instruction = f"{pdf_instruction}\n\nMANUAL RECHECK: This is a manual recheck request. Please be extremely thorough and consider all possible interpretations. Look for any mention, even indirect, of this attribute in the document context."
                            
                            manual_recheck_input = {
                                "context": context_text,
                                "extraction_instructions": manual_instruction,
                                "attribute_key": attribute_key,
                                "part_number": part_number if part_number else "Not Provided"
                            }
                            
                            json_result_str = loop.run_until_complete(
                                _invoke_chain_and_process(st.session_state.pdf_chain, manual_recheck_input, f"{attribute_key} (Manual Recheck)")
                            )
                            run_time = time.time() - start_time
                            
                            # Parse result
                            final_answer_value = "Error"
                            parse_error = None
                            raw_output = json_result_str if json_result_str else '{"error": "Manual recheck did not run"}'
                            
                            try:
                                string_to_parse = raw_output.strip()
                                parsed_json = extract_json_from_string(string_to_parse)
                                
                                if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                                    parsed_value = str(parsed_json[attribute_key])
                                    if (parsed_value.strip() == "" or 
                                        "not found" in parsed_value.lower() or
                                        parsed_value.lower() in ["none", "null", "n/a", "na"]):
                                        final_answer_value = "NOT FOUND (Manual)"
                                    else:
                                        final_answer_value = parsed_value
                                        st.success(f"Manual recheck successful for '{attribute_key}': {parsed_value}")
                                elif isinstance(parsed_json, dict) and "error" in parsed_json:
                                    final_answer_value = f"Error: {parsed_json['error'][:100]}"
                                    parse_error = ValueError(f"Manual Recheck Error: {parsed_json['error']}")
                                else:
                                    final_answer_value = "Unexpected JSON Format (Manual)"
                                    parse_error = ValueError(f"Manual Recheck Unexpected JSON format")
                                    
                            except Exception as processing_exc:
                                parse_error = processing_exc
                                final_answer_value = "Processing Error (Manual)"
                            
                            # Update the result
                            for i, result in enumerate(st.session_state.evaluation_results):
                                if result.get('Prompt Name') == prompt_name:
                                    previous_latency = result.get('Latency (s)', 0.0)
                                    total_latency = previous_latency + round(run_time, 2)
                                    
                                    # Check if we should preserve the original value (rollback logic)
                                    original_value = result.get('Extracted Value', '')
                                    original_source = result.get('Source', 'Unknown')
                                    
                                    # Rollback conditions: preserve original value if manual recheck failed
                                    should_rollback = (
                                        # Preserve "none" values when confirmed by recheck
                                        (original_value.lower() in ["none", "null", "n/a", "na"] and final_answer_value == "NOT FOUND (Manual)") or
                                        # Rollback to original when manual recheck has errors
                                        bool(parse_error) or
                                        final_answer_value in ["Error", "Processing Error (Manual)", "Unexpected JSON Format (Manual)"]
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
                                        final_source = 'Manual Recheck'
                                        is_success = not bool(parse_error) and final_answer_value not in ["NOT FOUND (Manual)", "Error", "Processing Error (Manual)", "Unexpected JSON Format (Manual)"]
                                        is_not_found = final_answer_value in ["NOT FOUND (Manual)"]
                                        is_error = bool(parse_error)
                                    
                                    st.session_state.evaluation_results[i].update({
                                        'Extracted Value': final_display_value,
                                        'Source': final_source,
                                        'Raw Output': raw_output if not should_rollback else result.get('Raw Output', raw_output),
                                        'Parse Error': str(parse_error) if parse_error and not should_rollback else result.get('Parse Error'),
                                        'Is Success': is_success,
                                        'Is Error': is_error,
                                        'Is Not Found': is_not_found,
                                        'Is Rate Limit': False,
                                        'Latency (s)': total_latency
                                    })
                                    
                                    # Show feedback for rollback
                                    if should_rollback and bool(parse_error):
                                        st.warning(f"⚠️ Rolled back to original '{original_value}' for '{attribute_key}' (manual recheck error)")
                                    elif should_rollback and original_value.lower() in ["none", "null", "n/a", "na"]:
                                        st.info(f"✅ Preserved original '{original_value}' for '{attribute_key}' (confirmed by manual recheck)")
                                    break
                            
                            time.sleep(0.5)  # Brief delay between manual rechecks
                            
                        except Exception as e:
                            st.error(f"Error during manual recheck for '{attribute_key}': {e}")
                            logger.error(f"Manual recheck failed for '{attribute_key}': {e}", exc_info=True)
                
                st.success("Manual recheck completed!")
                st.rerun()  # Refresh to show updated results
        else:
            st.success("All attributes have been successfully extracted! No manual recheck needed.")

        # --- View Raw Mistral Extraction ---
        if st.session_state.processed_documents:
            st.divider()
            st.header("4. View Raw Mistral Extraction")
            
            if st.button("👁️ View Raw Extracted Document Content", key="view_raw_extraction"):
                st.session_state.show_raw_extraction = not st.session_state.get('show_raw_extraction', False)
            
            if st.session_state.get('show_raw_extraction', False):
                st.info("This shows the raw document content extracted by Mistral Vision from your PDF pages.")
                
                # Group documents by source file
                docs_by_source = {}
                for doc in st.session_state.processed_documents:
                    source = doc.metadata.get('source', 'Unknown')
                    if source not in docs_by_source:
                        docs_by_source[source] = []
                    docs_by_source[source].append(doc)
                
                # Display documents grouped by source
                for source, docs in docs_by_source.items():
                    st.subheader(f"📄 {source}")
                    
                    # Sort documents by page number
                    docs.sort(key=lambda x: x.metadata.get('page', 0))
                    
                    for doc in docs:
                        page_num = doc.metadata.get('page', 'Unknown')
                        st.markdown(f"**Page {page_num}:**")
                        
                        # Display the raw content with syntax highlighting
                        st.code(doc.page_content, language='markdown')
                        
                        # Show metadata
                        with st.expander(f"Page {page_num} Metadata"):
                            st.json(doc.metadata)
                        
                        st.divider()
                
                # Add download button for raw extraction
                if st.button("📥 Download Raw Extraction (Markdown)", key="download_raw_extraction"):
                    # Create markdown content from all documents
                    raw_content = f"# Raw Mistral Extraction Report\n\n"
                    raw_content += f"**Extraction Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    raw_content += f"**Total Documents:** {len(st.session_state.processed_documents)}\n\n"
                    
                    # Group documents by source file
                    docs_by_source = {}
                    for doc in st.session_state.processed_documents:
                        source = doc.metadata.get('source', 'Unknown')
                        if source not in docs_by_source:
                            docs_by_source[source] = []
                        docs_by_source[source].append(doc)
                    
                    for source, docs in docs_by_source.items():
                        raw_content += f"## 📄 {source}\n\n"
                        
                        # Sort documents by page number
                        docs.sort(key=lambda x: x.metadata.get('page', 0))
                        
                        for doc in docs:
                            page_num = doc.metadata.get('page', 'Unknown')
                            raw_content += f"### Page {page_num}\n\n"
                            raw_content += f"**Metadata:**\n```json\n{json.dumps(doc.metadata, indent=2)}\n```\n\n"
                            raw_content += f"**Content:**\n```markdown\n{doc.page_content}\n```\n\n"
                            raw_content += "---\n\n"
                    
                    # Create download button
                    st.download_button(
                        label="📄 Download Raw Extraction Report",
                        data=raw_content.encode('utf-8'),
                        file_name='raw_mistral_extraction.md',
                        mime='text/markdown',
                        key='download_raw_md'
                    )

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
    elif (st.session_state.pdf_chain or st.session_state.web_chain) and st.session_state.extraction_performed:
        st.warning("Extraction process completed, but no valid results were generated for some fields. Check logs or raw outputs if available.")
with right_col:
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
    
    # Display extraction results in a beautiful format
    if st.session_state.evaluation_results:
        st.markdown("""
            <div class="extraction-results">
        """, unsafe_allow_html=True)
        
        # Create a summary of extracted data
        extracted_data = {}
        for result in st.session_state.evaluation_results:
            if isinstance(result, dict):
                prompt_name = result.get('Prompt Name', 'Unknown')
                extracted_value = result.get('Extracted Value', '')
                if extracted_value and extracted_value != 'NOT FOUND' and extracted_value != 'ERROR':
                    extracted_data[prompt_name] = extracted_value
        
        # Display each extracted item beautifully
        for key, value in extracted_data.items():
            # Truncate long values for better display
            display_value = value[:100] + "..." if len(value) > 100 else value
            st.markdown(f"""
                <div class="result-item">
                    <div class="result-label">🔍 {key}</div>
                    <div class="result-value" title="{value}">{display_value}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Success metrics
        if st.session_state.evaluation_metrics:
            metrics = st.session_state.evaluation_metrics
            st.markdown("""
                <div style="background: white; border-radius: 15px; padding: 1rem; margin: 1rem 0; box-shadow: 0 4px 15px rgba(30, 60, 114, 0.1);">
                    <h4 style="color: #1e3c72; margin-bottom: 1rem;">📈 Success Metrics</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                success_rate = metrics.get('success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1%}")
            with col2:
                total_fields = metrics.get('total_fields', 0)
                st.metric("Total Fields", total_fields)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("📄 Upload and process documents to see extracted results here.")
    
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
    