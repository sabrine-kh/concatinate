# --- Force python to use pysqlite3 based on chromadb docs ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

import streamlit as st
import os
import time
from loguru import logger
import json
import pandas as pd
import re
import asyncio
import subprocess
import nest_asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx

nest_asyncio.apply()

st.markdown(
    """<style>
    [data-testid='stSidebarNav'] {display: none;}
    </style>""",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
    if st.button("🏠 Home"):
        st.switch_page("app.py")
    if st.button("💬 Chat with Leoparts"):
        st.switch_page("pages/chatbot.py")
    if st.button("📄 Extract a new Part"):
        st.switch_page("pages/extraction_attributs.py")

def install_playwright_browsers():
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

import config
from pdf_processor import process_uploaded_pdfs
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
from llm_interface import (
    initialize_llm,
    create_pdf_extraction_chain,
    create_web_extraction_chain,
    _invoke_chain_and_process,
    scrape_website_table_html
)
from extraction_prompts import (
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,
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
    WORKING_TEMPERATURE_PROMPT,
    HOUSING_SEAL_PROMPT,
    WIRE_SEAL_PROMPT,
    SEALING_PROMPT,
    SEALING_CLASS_PROMPT,
    CONTACT_SYSTEMS_PROMPT,
    TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT,
    CLOSED_CAVITIES_PROMPT,
    PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT,
    SET_KIT_PROMPT,
    HV_QUALIFIED_PROMPT
)
from extraction_prompts_web import (
    MATERIAL_FILLING_WEB_PROMPT,
    MATERIAL_NAME_WEB_PROMPT,
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
    MAX_WORKING_TEMPERATURE_WEB_PROMPT,
    MIN_WORKING_TEMPERATURE_WEB_PROMPT,
    HOUSING_SEAL_WEB_PROMPT,
    WIRE_SEAL_WEB_PROMPT,
    SEALING_WEB_PROMPT,
    SEALING_CLASS_WEB_PROMPT,
    CONTACT_SYSTEMS_WEB_PROMPT,
    TERMINAL_POSITION_ASSURANCE_WEB_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_WEB_PROMPT,
    CLOSED_CAVITIES_WEB_PROMPT,
    PRE_ASSEMBLED_WEB_PROMPT,
    CONNECTOR_TYPE_WEB_PROMPT,
    SET_KIT_WEB_PROMPT,
    HV_QUALIFIED_WEB_PROMPT
)

@st.cache_resource
def initialize_embeddings():
    try:
        embedding_function = get_embedding_function()
        return embedding_function
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        return None

@st.cache_resource
def initialize_llm_cached():
    try:
        llm = initialize_llm()
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        return None

def main():
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
         if not st.exception:
            st.error("Core components (Embeddings or LLM) failed to initialize. Cannot continue.")
         st.stop()
    if st.session_state.retriever is None and config.CHROMA_SETTINGS.is_persistent and embedding_function:
        logger.info("Attempting to load existing vector store...")
        st.session_state.retriever = load_existing_vector_store(embedding_function)
        if st.session_state.retriever:
            logger.success("Successfully loaded retriever from persistent storage.")
            st.session_state.processed_files = ["Existing data loaded from disk"]
            logger.info("Creating extraction chains from loaded retriever...")
            st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
            st.session_state.web_chain = create_web_extraction_chain(llm)
            if not st.session_state.pdf_chain or not st.session_state.web_chain:
                st.warning("Failed to create one or both extraction chains from loaded retriever.")
            st.session_state.extraction_performed = False
        else:
            logger.warning("No existing persistent vector store found or failed to load.")
    with st.sidebar:
        st.header("1. Document Processing")
        if st.button("🤖 Open Chatbot in New Page", type="primary", use_container_width=True):
            st.switch_page("pages/chatbot.py")
        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        st.text_input("Enter Part Number (Optional):", key="part_number_input", value=st.session_state.get("part_number_input", ""))
        process_button = st.button("Process Uploaded Documents", key="process_button", type="primary")
        if process_button and uploaded_files:
            if not embedding_function or not llm:
                 st.error("Core components (Embeddings or LLM) failed to initialize earlier. Cannot process documents.")
            else:
                st.session_state.retriever = None
                st.session_state.pdf_chain = None
                st.session_state.web_chain = None
                st.session_state.processed_files = []
                st.session_state.evaluation_results = []
                st.session_state.extraction_performed = False
                st.session_state.scraped_table_html_cache = None
                st.session_state.current_part_number_scraped = None
                if 'gt_editor' in st.session_state:
                    del st.session_state['gt_editor']
                filenames = [f.name for f in uploaded_files]
                logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)}")
                with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
                    processed_docs = None
                    try:
                        start_time = time.time()
                        temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                        processed_docs = process_uploaded_pdfs(uploaded_files, temp_dir)
                        processing_time = time.time() - start_time
                        logger.info(f"PDF processing took {processing_time:.2f} seconds.")
                    except Exception as e:
                        logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                        st.error(f"Error processing PDFs: {e}")
                if processed_docs:
                    logger.info(f"Generated {len(processed_docs)} document chunks.")
                    with st.spinner("Indexing documents in vector store..."):
                        try:
                            start_time = time.time()
                            st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                            indexing_time = time.time() - start_time
                            logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")
                            if st.session_state.retriever:
                                st.session_state.processed_files = filenames
                                logger.success("Vector store setup complete. Retriever is ready.")
                                with st.spinner("Preparing extraction engines..."):
                                     st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                                     st.session_state.web_chain = create_web_extraction_chain(llm)
                                if st.session_state.pdf_chain and st.session_state.web_chain:
                                    logger.success("Extraction chains created.")
                                    st.success(f"Successfully processed {len(filenames)} file(s). Evaluation below.")
                                else:
                                    st.error("Failed to create one or both extraction chains after processing.")
                            else:
                                st.error("Failed to setup vector store after processing PDFs.")
                        except Exception as e:
                            logger.error(f"Error setting up vector store: {e}", exc_info=True)
                            st.error(f"Error setting up vector store: {e}")
                else:
                    st.warning("No text could be extracted or processed from the uploaded PDFs.")
        elif process_button:
            st.warning("Please upload at least one PDF file before processing.")
    st.subheader("Processing Status")
    if st.session_state.pdf_chain and st.session_state.web_chain and st.session_state.processed_files:
        st.success(f"Ready. Processed: {', '.join(st.session_state.processed_files)}")
    elif config.CHROMA_SETTINGS.is_persistent and st.session_state.retriever and (not st.session_state.pdf_chain or not st.session_state.web_chain):
        st.warning("Loaded existing data, but failed to create one or both extraction chains.")
    elif config.CHROMA_SETTINGS.is_persistent and st.session_state.retriever:
        st.success(f"Ready. Using existing data loaded from disk.")
    else:
        st.info("Upload and process PDF documents to view extracted data.")
    st.header("2. Extracted Information")
    st.write("DEBUG - processed_files:", st.session_state.get('processed_files'))
    st.write("DEBUG - pdf_chain:", st.session_state.get('pdf_chain'))
    st.write("DEBUG - web_chain:", st.session_state.get('web_chain'))
    st.write("DEBUG - evaluation_results:", st.session_state.get('evaluation_results'))
    if not st.session_state.pdf_chain or not st.session_state.web_chain:
        st.info("Upload and process documents using the sidebar to see extracted results here.")
        return

    # --- Exemple d'extraction d'attribut (MATERIAL_NAME) ---
    if st.session_state.pdf_chain:
        extraction_input = {
            "attribute_key": "MATERIAL NAME",
            "extraction_instructions": {"extraction_instructions": MATERIAL_NAME_PROMPT},
            "part_number": {"part_number": st.session_state.get("part_number_input", "")}
        }
        result = asyncio.run(_invoke_chain_and_process(st.session_state.pdf_chain, extraction_input, "MATERIAL NAME"))
        st.session_state.evaluation_results = [result]

    # --- Affichage du résultat ---
    if st.session_state.evaluation_results:
        st.subheader("Résultat d'extraction (exemple) :")
        for res in st.session_state.evaluation_results:
            st.write(res)

if __name__ == "__main__":
    main() 