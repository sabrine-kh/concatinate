# --- Force python to use pysqlite3 based on chromadb docs ---
# This override MUST happen before any other imports that might import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# app.py
import streamlit as st
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

install_playwright_browsers() # Run the installation check on script start
# ----------------------------------------------------

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


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PDF Auto-Extraction with Groq",
    page_icon="📄",
    layout="wide"
)

# Create tabs for navigation
tab1, tab2 = st.tabs(["📄 PDF Extraction", "🤖 Chatbot"])

# --- Logging Configuration ---
# Configure Loguru logger (can be more flexible than standard logging)
# logger.add("logs/app_{time}.log", rotation="10 MB", level="INFO") # Example: Keep file logging if desired
# Toasts are disabled as per previous request
# Errors will still be shown via st.error where used explicitly

# --- Application State ---
# Use Streamlit's session state to hold persistent data across reruns
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
# Add states for BOTH chains
if 'pdf_chain' not in st.session_state:
    st.session_state.pdf_chain = None
if 'web_chain' not in st.session_state:
    st.session_state.web_chain = None
# Remove old single chain state
# if 'extraction_chain' not in st.session_state:
#     st.session_state.extraction_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = [] # Store names of processed files
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = [] # List to store detailed results per field
if 'extraction_performed' not in st.session_state:
    st.session_state.extraction_performed = False
if 'scraped_table_html_cache' not in st.session_state:
    st.session_state.scraped_table_html_cache = None # Cache for scraped HTML for the current part number
if 'current_part_number_scraped' not in st.session_state:
    st.session_state.current_part_number_scraped = None # Track which part number was last scraped for

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "extraction"

# --- Global Variables / Initialization ---
# Initialize embeddings (this is relatively heavy, do it once)
@st.cache_resource
def initialize_embeddings():
    # Let exceptions from get_embedding_function propagate
    embeddings = get_embedding_function()
    return embeddings

# Initialize LLM (also potentially heavy/needs API key check)
@st.cache_resource
def initialize_llm_cached():
    # logger.info("Attempting to initialize LLM...") # Log before calling if needed
    llm_instance = initialize_llm()
    # logger.success("LLM initialized successfully.") # Log after successful call if needed
    return llm_instance

# --- Wrap the cached function call in try-except ---
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

# --- Check if initializations failed ---
if embedding_function is None or llm is None:
     if not st.exception: # If st.stop() wasn't called already
        st.error("Core components (Embeddings or LLM) failed to initialize. Cannot continue.")
     st.stop()


# --- Load existing vector store or process uploads ---
# Reset evaluation state when processing new files
def reset_evaluation_state():
    st.session_state.evaluation_results = []
    st.session_state.extraction_performed = False # Reset the flag here too
    st.session_state.scraped_table_html_cache = None # Clear scraped HTML cache
    st.session_state.current_part_number_scraped = None # Clear scraped part number tracker
    # Clear data editor state if it exists
    if 'gt_editor' in st.session_state:
        del st.session_state['gt_editor']

# Try loading existing vector store and create BOTH extraction chains
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
            reset_evaluation_state() # Reset evaluation results AND extraction_performed flag

            filenames = [f.name for f in uploaded_files]
            logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)}")
            # --- PDF Processing ---
            with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
                processed_docs = None # Initialize
                try:
                    start_time = time.time()
                    temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                    processed_docs = process_uploaded_pdfs(uploaded_files, temp_dir)
                    processing_time = time.time() - start_time
                    logger.info(f"PDF processing took {processing_time:.2f} seconds.")
                except Exception as e:
                    logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                    st.error(f"Error processing PDFs: {e}")

            # --- Vector Store Indexing ---
            if processed_docs:
                logger.info(f"Generated {len(processed_docs)} document chunks.")
                with st.spinner("Indexing documents in vector store..."):
                    try:
                        start_time = time.time()
                        st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                        indexing_time = time.time() - start_time
                        logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")

                        if st.session_state.retriever:
                            st.session_state.processed_files = filenames # Update list
                            logger.success("Vector store setup complete. Retriever is ready.")
                            # --- Create BOTH Extraction Chains --- 
                            with st.spinner("Preparing extraction engines..."):
                                 st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                                 st.session_state.web_chain = create_web_extraction_chain(llm)
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
    elif config.CHROMA_SETTINGS.is_persistent and st.session_state.retriever and (not st.session_state.pdf_chain or not st.session_state.web_chain):
         st.warning("Loaded existing data, but failed to create one or both extraction chains.")
    elif config.CHROMA_SETTINGS.is_persistent and st.session_state.retriever:
         st.success(f"Ready. Using existing data loaded from disk.") # Assuming chains created on load
    else:
        st.info("Upload and process PDF documents to view extracted data.")


# --- Main Area for Displaying Extraction Results ---
st.header("2. Extracted Information")

# --- Event loop management ---
def get_or_create_eventloop():
    """Get the current event loop or create a new one"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def run_async_with_retry(coro):
    """Run async code with retry logic in case of loop shutdown"""
    try:
        loop = get_or_create_eventloop()
        return loop.run_until_complete(coro)
    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e):
            # Create new loop and retry
            loop = get_or_create_eventloop()
            return loop.run_until_complete(coro)
        raise

# Initialize the event loop at startup
get_or_create_eventloop()

# Check if BOTH chains are ready before proceeding
if not st.session_state.pdf_chain or not st.session_state.web_chain:
    st.info("Upload and process documents using the sidebar to see extracted results here.")
    # Ensure evaluation state is also clear if no chain
    if not st.session_state.evaluation_results and not st.session_state.extraction_performed:
         reset_evaluation_state() # Ensure reset if no chain and extraction not done
else:
    # --- Block 1: Run Extraction (if needed) --- 
    if (st.session_state.pdf_chain and st.session_state.web_chain) and not st.session_state.extraction_performed:
        # --- Get Part Number --- 
        part_number = st.session_state.get("part_number_input", "").strip()
        # ---------------------

        # Define the prompts (attribute keys mapped to PDF and WEB instructions)
        prompts_to_run = { 
            # Material Properties
            "Material Filling": {"pdf": MATERIAL_PROMPT, "web": MATERIAL_FILLING_WEB_PROMPT},
            "Material Name": {"pdf": MATERIAL_NAME_PROMPT, "web": MATERIAL_NAME_WEB_PROMPT},
            # Physical / Mechanical Attributes
            "Pull-to-Seat": {"pdf": PULL_TO_SEAT_PROMPT, "web": PULL_TO_SEAT_WEB_PROMPT},
            "Gender": {"pdf": GENDER_PROMPT, "web": GENDER_WEB_PROMPT},
            "Height [MM]": {"pdf": HEIGHT_MM_PROMPT, "web": HEIGHT_MM_WEB_PROMPT},
            "Length [MM]": {"pdf": LENGTH_MM_PROMPT, "web": LENGTH_MM_WEB_PROMPT},
            "Width [MM]": {"pdf": WIDTH_MM_PROMPT, "web": WIDTH_MM_WEB_PROMPT},
            "Number of Cavities": {"pdf": NUMBER_OF_CAVITIES_PROMPT, "web": NUMBER_OF_CAVITIES_WEB_PROMPT},
            "Number of Rows": {"pdf": NUMBER_OF_ROWS_PROMPT, "web": NUMBER_OF_ROWS_WEB_PROMPT},
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
            "Closed Cavities": {"pdf": CLOSED_CAVITIES_PROMPT, "web": CLOSED_CAVITIES_WEB_PROMPT},
            # Assembly & Type
            "Pre-Assembled": {"pdf": PRE_ASSEMBLED_PROMPT, "web": PRE_ASSEMBLED_WEB_PROMPT},
            "Type of Connector": {"pdf": CONNECTOR_TYPE_PROMPT, "web": CONNECTOR_TYPE_WEB_PROMPT},
            "Set/Kit": {"pdf": SET_KIT_PROMPT, "web": SET_KIT_WEB_PROMPT},
            # Specialized Attributes
            "HV Qualified": {"pdf": HV_QUALIFIED_PROMPT, "web": HV_QUALIFIED_WEB_PROMPT}
        }

        # Initialize results storage
        intermediate_results = {}  # Store stage 1 results {prompt_name: {result_data}}
        pdf_fallback_needed = []   # List of prompt_names needing stage 2

        # --- Block 1a: Scrape Web Table HTML (if needed) --- 
        scraped_table_html = None # Initialize
        if part_number: # Only scrape if part number is provided
            # Check cache first
            if st.session_state.current_part_number_scraped == part_number and st.session_state.scraped_table_html_cache is not None:
                 logger.info(f"Using cached scraped HTML for part number {part_number}.")
                 scraped_table_html = st.session_state.scraped_table_html_cache
            else:
                 # Scrape and update cache
                 logger.info(f"Part number {part_number} changed or not cached. Attempting web scrape...")
                 with st.spinner("Attempting to scrape data from supplier websites..."):
                     scrape_start_time = time.time()
                     try:
                          # Ensure scrape_website_table_html is imported from llm_interface
                          from llm_interface import scrape_website_table_html
                          # Create new event loop for scraping
                          loop = asyncio.new_event_loop()
                          asyncio.set_event_loop(loop)
                          scraped_table_html = loop.run_until_complete(scrape_website_table_html(part_number))
                          loop.close()
                          scrape_time = time.time() - scrape_start_time
                          if scraped_table_html:
                              logger.success(f"Web scraping successful in {scrape_time:.2f} seconds.")
                              st.caption(f"ℹ️ Found web data for part# {part_number}. Will prioritize.")
                          else:
                              logger.warning(f"Web scraping attempted but failed to find table HTML in {scrape_time:.2f} seconds.")
                              st.caption(f"⚠️ Web scraping failed for part# {part_number}, using PDF data only.")
                          # Update cache
                          st.session_state.scraped_table_html_cache = scraped_table_html
                          st.session_state.current_part_number_scraped = part_number
                     except Exception as scrape_e:
                          scrape_time = time.time() - scrape_start_time
                          logger.error(f"Error during web scraping ({scrape_time:.2f}s): {scrape_e}", exc_info=True)
                          st.warning(f"An error occurred during web scraping: {scrape_e}. Using PDF data only.")
                          # Ensure cache is cleared on error
                          st.session_state.scraped_table_html_cache = None
                          st.session_state.current_part_number_scraped = part_number
        else:
             logger.info("No part number provided, skipping web scrape.")
             # Clear cache if part number is removed
             if st.session_state.current_part_number_scraped is not None:
                  st.session_state.scraped_table_html_cache = None
                  st.session_state.current_part_number_scraped = None

        # --- Stage 1: Web Extraction --- 
        if scraped_table_html:
            # Initialize columns for displaying progress
            cols = st.columns(2)
            col_index = 0
            SLEEP_INTERVAL_SECONDS = 0.2 # Can potentially be lower for web chain

            # Process each prompt
            for prompt_name, instructions in prompts_to_run.items():
                attribute_key = prompt_name
                web_instruction = instructions["web"]
                
                # Get current column for display
                current_col = cols[col_index % 2]
                col_index += 1
                
                # Initialize variables for this iteration
                json_result_str = None
                run_time = 0.0
                source = "Web"
                
                with current_col:
                     with st.spinner(f"Stage 1: Extracting {attribute_key} from Web Data..."):
                        try:
                            start_time = time.time()
                            web_input = {
                                "cleaned_web_data": scraped_table_html,
                                "attribute_key": attribute_key,
                                "extraction_instructions": web_instruction
                            }
                            # Create new event loop for each extraction
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            json_result_str = loop.run_until_complete(
                                _invoke_chain_and_process(st.session_state.web_chain, web_input, f"{attribute_key} (Web)")
                            )
                            loop.close()
                            run_time = time.time() - start_time
                            logger.info(f"Stage 1 (Web) for '{attribute_key}' took {run_time:.2f} seconds.")
                            time.sleep(SLEEP_INTERVAL_SECONDS)
                        except Exception as e:
                            logger.error(f"Error during Stage 1 (Web) call for '{attribute_key}': {e}", exc_info=True)
                            json_result_str = f'{{"error": "Exception during Stage 1 call: {e}"}}'
                            run_time = time.time() - start_time

                # --- Basic Parsing of Stage 1 Result --- 
                final_answer_value = "Error"
                parse_error = None
                is_rate_limit = False
                llm_returned_error_msg = None
                raw_output = json_result_str if json_result_str else '{"error": "Stage 1 did not run"}'
                needs_fallback = False
                
                try:
                    # Minimal cleaning, rely on helper's cleaning primarily
                    string_to_parse = raw_output.strip()
                    parsed_json = json.loads(string_to_parse)
                    
                    if isinstance(parsed_json, dict):
                        if attribute_key in parsed_json:
                             parsed_value = str(parsed_json[attribute_key])
                             # Check for NOT FOUND variants 
                             if "not found" in parsed_value.lower() or parsed_value.strip() == "":
                                 final_answer_value = "NOT FOUND"
                                 needs_fallback = True # Mark for PDF stage
                                 logger.info(f"Stage 1 result for '{attribute_key}' is NOT FOUND. Queued for PDF fallback.")
                             else:
                                 final_answer_value = parsed_value # Store successful web result
                                 logger.success(f"Stage 1 successful for '{attribute_key}' from Web data.")
                        elif "error" in parsed_json:
                            # Handle errors from the web chain call
                            error_msg = parsed_json['error']
                            llm_returned_error_msg = error_msg
                            if "rate limit" in error_msg.lower():
                                final_answer_value = "Rate Limit Hit"
                                is_rate_limit = True
                                parse_error = ValueError("Rate limit hit (Web)")
                            else:
                                final_answer_value = f"Error: {error_msg[:100]}"
                                parse_error = ValueError(f"Stage 1 Error: {error_msg}")
                            needs_fallback = True # Also fallback on web chain error
                            logger.warning(f"Stage 1 Error for '{attribute_key}'. Queued for PDF fallback. Error: {error_msg}")
                        else:
                             final_answer_value = "Unexpected JSON Format"
                             parse_error = ValueError(f"Stage 1 Unexpected JSON keys: {list(parsed_json.keys())}")
                             needs_fallback = True # Fallback on unexpected format
                    else:
                        final_answer_value = "Unexpected JSON Type"
                        parse_error = TypeError(f"Stage 1 Expected dict, got {type(parsed_json)}")
                        needs_fallback = True # Fallback
                        logger.warning(f"Stage 1 Unexpected JSON type '{attribute_key}'. Queued for PDF fallback.")
                        
                except json.JSONDecodeError as json_err:
                    parse_error = json_err
                    final_answer_value = "Invalid JSON Response"
                    logger.error(f"Failed to parse Stage 1 JSON for '{attribute_key}'. Error: {json_err}. String: '{string_to_parse}'")
                    needs_fallback = True # Fallback on bad JSON
                except Exception as processing_exc:
                    parse_error = processing_exc
                    final_answer_value = "Processing Error"
                    logger.error(f"Error processing Stage 1 result for '{attribute_key}'. Error: {processing_exc}")
                    needs_fallback = True # Fallback
                
                # Store intermediate result (even if NOT FOUND or error)
                is_error = bool(parse_error) and not is_rate_limit
                is_not_found_stage1 = final_answer_value == "NOT FOUND"
                is_success_stage1 = not is_error and not is_not_found_stage1 and not is_rate_limit
                
                intermediate_results[prompt_name] = {
                    'Prompt Name': prompt_name,
                    'Extracted Value': final_answer_value, # Store Stage 1 value/error/NOT FOUND
                    'Ground Truth': '',
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
                
                if needs_fallback:
                    pdf_fallback_needed.append(prompt_name)

        else: # No scraped HTML, all attributes need PDF fallback
            logger.info("No scraped web data available. All attributes will use PDF extraction.")
            pdf_fallback_needed = list(prompts_to_run.keys())
            # Populate intermediate results with placeholders indicating skipped web stage
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

        # --- Stage 2: PDF Fallback --- 
        st.info(f"Running Stage 2 (PDF Fallback) for {len(pdf_fallback_needed)} attributes...")
        col_index = 0 # Reset column index
        SLEEP_INTERVAL_SECONDS = 0.5 # Potentially longer delay for more complex chain

        if not pdf_fallback_needed:
            st.success("Stage 1 extraction successful for all attributes from web data.")
        else:
            # Initialize columns for Stage 2
            cols = st.columns(2)
            
            for prompt_name in pdf_fallback_needed:
                attribute_key = prompt_name  # Initialize attribute_key from prompt_name
                pdf_instruction = prompts_to_run[attribute_key]["pdf"] # Get specific PDF instruction
                current_col = cols[col_index % 2]
                col_index += 1
                
                # Initialize variables for this iteration
                json_result_str = None
                run_time = 0.0
                source = "PDF" # Source for this stage
                
                with current_col:
                     with st.spinner(f"Stage 2: Extracting {attribute_key} from PDF Data..."):
                        try:
                            start_time = time.time()
                            pdf_input = {
                                "extraction_instructions": pdf_instruction, # Use specific PDF instruction
                                "attribute_key": attribute_key,
                                "part_number": part_number if part_number else "Not Provided"
                            }
                            # Create new event loop for each extraction
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            json_result_str = loop.run_until_complete(
                                _invoke_chain_and_process(st.session_state.pdf_chain, pdf_input, f"{attribute_key} (PDF)")
                            )
                            loop.close()
                            run_time = time.time() - start_time
                            logger.info(f"Stage 2 (PDF) for '{attribute_key}' took {run_time:.2f} seconds.")
                            time.sleep(SLEEP_INTERVAL_SECONDS)
                        except Exception as e:
                             logger.error(f"Error during Stage 2 (PDF) call for '{attribute_key}': {e}", exc_info=True)
                             json_result_str = f'{{"error": "Exception during Stage 2 call: {e}"}}'
                             run_time = time.time() - start_time
                
                # --- Basic Parsing of Stage 2 Result --- 
                final_answer_value = "Error"
                parse_error = None
                is_rate_limit = False
                llm_returned_error_msg = None
                raw_output = json_result_str if json_result_str else '{"error": "Stage 2 did not run"}'
                
                try:
                    string_to_parse = raw_output.strip()
                    parsed_json = json.loads(string_to_parse)
                    if isinstance(parsed_json, dict):
                        if attribute_key in parsed_json:
                            final_answer_value = str(parsed_json[attribute_key]) # Store final PDF result
                            logger.success(f"Stage 2 successful for '{attribute_key}' from PDF data.")
                        elif "error" in parsed_json:
                            error_msg = parsed_json['error']
                            llm_returned_error_msg = error_msg
                            if "rate limit" in error_msg.lower():
                                final_answer_value = "Rate Limit Hit"
                                is_rate_limit = True
                                parse_error = ValueError("Rate limit hit (PDF)")
                            else:
                                final_answer_value = f"Error: {error_msg[:100]}"
                                parse_error = ValueError(f"Stage 2 Error: {error_msg}")
                            logger.warning(f"Stage 2 Error for '{attribute_key}' from PDF. Error: {error_msg}")
                        else:
                             final_answer_value = "Unexpected JSON Format"
                             parse_error = ValueError(f"Stage 2 Unexpected JSON keys: {list(parsed_json.keys())}")
                             logger.warning(f"Stage 2 Unexpected JSON for '{attribute_key}'.")
                    else:
                         final_answer_value = "Unexpected JSON Type"
                         parse_error = TypeError(f"Stage 2 Expected dict, got {type(parsed_json)}")
                         logger.warning(f"Stage 2 Unexpected JSON type for '{attribute_key}'.")
                         
                except json.JSONDecodeError as json_err:
                    parse_error = json_err
                    final_answer_value = "Invalid JSON Response"
                    logger.error(f"Failed to parse Stage 2 JSON for '{attribute_key}'. Error: {json_err}. String: '{string_to_parse}'")
                except Exception as processing_exc:
                    parse_error = processing_exc
                    final_answer_value = "Processing Error"
                    logger.error(f"Error processing Stage 2 result for '{attribute_key}'. Error: {processing_exc}")

                # --- Update the result in intermediate_results with Stage 2 data --- 
                is_error = bool(parse_error) and not is_rate_limit
                is_not_found_stage2 = "not found" in final_answer_value.lower() or final_answer_value.strip() == ""
                is_success_stage2 = not is_error and not is_not_found_stage2 and not is_rate_limit
                
                # Add Stage 2 latency to existing Stage 1 latency if Stage 1 ran
                stage1_latency = intermediate_results[prompt_name].get('Latency (s)', 0.0)
                total_latency = stage1_latency + round(run_time, 2)
                
                intermediate_results[prompt_name].update({
                    'Extracted Value': final_answer_value, # OVERWRITE with Stage 2 value/error
                    'Source': source, # Update source to PDF
                    'Raw Output': raw_output, # Store Stage 2 raw output
                    'Parse Error': str(parse_error) if parse_error else None,
                    'Is Success': is_success_stage2,
                    'Is Error': is_error,
                    'Is Not Found': is_not_found_stage2,
                    'Is Rate Limit': is_rate_limit,
                    'Latency (s)': total_latency 
                })
                logger.info(f"Updated result for '{prompt_name}' with PDF fallback data.")

        # --- Final Processing --- 
        # Convert intermediate_results dict to list
        extraction_results_list = list(intermediate_results.values()) 
        
        # Set extraction_performed flag and handle success/error messages
        extraction_successful = True # Assume success unless critical errors occurred (e.g., chain init)

        if extraction_successful:
            st.session_state.evaluation_results = extraction_results_list
            st.session_state.extraction_performed = True
            st.success("Extraction complete (using Web data where possible, falling back to PDF). Enter ground truth below.")
            # st.rerun() # REMOVE/COMMENT OUT to keep cards visible
        else:
            st.error("Extraction process encountered critical issues.")
            # Optionally store partial results if desired
            st.session_state.evaluation_results = extraction_results_list
            st.session_state.extraction_performed = True
            # st.rerun() # REMOVE/COMMENT OUT to keep cards visible (even on error)


    # --- Block 2: Display Results (if they exist) ---
    if st.session_state.evaluation_results:
        st.divider()
        st.header("3. Extraction Results")

        results_df = pd.DataFrame(st.session_state.evaluation_results)
        if 'Source' not in results_df.columns:
             results_df['Source'] = 'Unknown' # Add placeholder if missing

        # Display results without metrics
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prompt Name": st.column_config.TextColumn(width="medium"),
                "Extracted Value": st.column_config.TextColumn(width="medium"),
                "Source": st.column_config.TextColumn(width="small"),
                "Raw Output": st.column_config.TextColumn("Raw Output", width="large"),
                "Parse Error": st.column_config.TextColumn("Parse Error", width="medium")
            }
        )

        # --- Export Section --- 
        st.divider()
        st.header("4. Export Results")

        # Prepare data for export
        export_df = pd.DataFrame(st.session_state.evaluation_results)

        # Convert DataFrame to CSV
        @st.cache_data # Cache the conversion
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(export_df)

        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name='extraction_results.csv',
            mime='text/csv',
            key='download_csv'
        )

    # --- Block 3: Handle cases where extraction ran but yielded nothing, or hasn't run ---
    elif (st.session_state.pdf_chain or st.session_state.web_chain) and st.session_state.extraction_performed:
        st.warning("Extraction process completed, but no valid results were generated for some fields. Check logs or raw outputs if available.")
    

# REMOVE the previous Q&A section entirely (already done)

# In the main section, use asyncio.run to run the async function
if (st.session_state.pdf_chain and st.session_state.web_chain) and not st.session_state.extraction_performed:
    asyncio.run(process_extraction())

# Main content area with conditional rendering
if st.session_state.current_page == "extraction":
    # --- UI Layout ---
    persistence_enabled = config.CHROMA_SETTINGS.is_persistent
    st.title("📄 PDF Auto-Extraction with Groq")
    st.markdown("Upload PDF documents, process them, and view automatically extracted information.")
    st.markdown(f"**Model:** `{config.LLM_MODEL_NAME}` | **Embeddings:** `{config.EMBEDDING_MODEL_NAME}` | **Persistence:** `{'Enabled' if persistence_enabled else 'Disabled'}`")

    # Check for API Key
    if not config.GROQ_API_KEY:
        st.warning("Groq API Key not found. Please set the GROQ_API_KEY environment variable.", icon="⚠️")

    # Run extraction if needed
    if (st.session_state.pdf_chain and st.session_state.web_chain) and not st.session_state.extraction_performed:
        asyncio.run(process_extraction())
    
elif st.session_state.current_page == "chatbot":
    # Import and run the chatbot code
    import pages.chatbot as chatbot
    chatbot.run_chatbot()