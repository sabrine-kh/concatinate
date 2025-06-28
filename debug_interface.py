import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from debug_logger import debug_logger, log_streamlit_state
import os

def parse_debug_log(log_content: str) -> list:
    """Parse the debug log content into structured data"""
    entries = []
    current_entry = {}
    lines = log_content.split('\n')
    
    for line in lines:
        if line.startswith('=' * 60):
            if current_entry:
                entries.append(current_entry)
                current_entry = {}
            continue
            
        if line.startswith('STEP '):
            # Parse step header
            parts = line.split(' - ')
            if len(parts) >= 3:
                step_info = parts[0].replace('STEP ', '')
                level = parts[1]
                timestamp = ' - '.join(parts[2:])
                current_entry = {
                    'step': step_info,
                    'level': level,
                    'timestamp': timestamp,
                    'message': '',
                    'context': {},
                    'data': None
                }
        elif line.startswith('MESSAGE: '):
            current_entry['message'] = line.replace('MESSAGE: ', '')
        elif line.startswith('CONTEXT: '):
            try:
                context_str = line.replace('CONTEXT: ', '')
                current_entry['context'] = json.loads(context_str)
            except:
                current_entry['context'] = {'raw': context_str}
        elif line.startswith('DATA: '):
            try:
                data_str = line.replace('DATA: ', '')
                current_entry['data'] = json.loads(data_str)
            except:
                current_entry['data'] = data_str
    
    if current_entry:
        entries.append(current_entry)
    
    return entries

def create_debug_interface():
    """Create the main debug interface"""
    st.set_page_config(page_title="Debug Interface", layout="wide")
    
    st.title("🔍 Debug Interface")
    st.markdown("Real-time debugging and logging interface for LEOPARTS")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Debug Controls")
        
        # Log file selection
        log_files = [f for f in os.listdir('.') if f.endswith('.txt') and 'debug' in f.lower()]
        selected_log = st.selectbox("Select Log File", log_files, index=0 if log_files else None)
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
        
        # Filter controls
        st.header("Filters")
        level_filter = st.multiselect(
            "Log Level",
            ["INFO", "WARNING", "ERROR", "DEBUG"],
            default=["INFO", "WARNING", "ERROR"]
        )
        
        message_filter = st.text_input("Message contains", "")
        
        # Action buttons
        st.header("Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Log"):
                debug_logger.clear_log()
                st.success("Log cleared!")
                st.rerun()
        
        # Log current session state
        if st.button("📊 Log Session State"):
            log_streamlit_state()
            st.success("Session state logged!")
    
    # Main content area
    if selected_log and os.path.exists(selected_log):
        # Read and parse log
        with open(selected_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        entries = parse_debug_log(log_content)
        
        if entries:
            # Create DataFrame
            df = pd.DataFrame(entries)
            
            # Apply filters
            if level_filter:
                df = df[df['level'].isin(level_filter)]
            
            if message_filter:
                df = df[df['message'].str.contains(message_filter, case=False, na=False)]
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Entries", len(entries))
            with col2:
                st.metric("Filtered Entries", len(df))
            with col3:
                error_count = len([e for e in entries if e.get('level') == 'ERROR'])
                st.metric("Errors", error_count)
            with col4:
                warning_count = len([e for e in entries if e.get('level') == 'WARNING'])
                st.metric("Warnings", warning_count)
            
            # Display entries
            st.subheader("Log Entries")
            
            # Entry viewer
            for idx, entry in df.iterrows():
                with st.expander(f"Step {entry['step']} - {entry['level']} - {entry['message'][:100]}..."):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Step:**", entry['step'])
                        st.write("**Level:**", entry['level'])
                        st.write("**Timestamp:**", entry['timestamp'])
                        st.write("**Message:**", entry['message'])
                    
                    with col2:
                        if entry.get('context'):
                            st.write("**Context:**")
                            st.json(entry['context'])
                        
                        if entry.get('data'):
                            st.write("**Data:**")
                            if isinstance(entry['data'], dict):
                                st.json(entry['data'])
                            else:
                                st.text(str(entry['data']))
            
            # Raw log view
            with st.expander("📄 Raw Log Content"):
                st.text_area("Raw Log", log_content, height=400)
        
        else:
            st.info("No log entries found or log file is empty.")
    
    else:
        st.warning("No debug log file found. Start using the debug logger to see entries here.")
        
        # Show usage instructions
        with st.expander("📖 How to use the debug logger"):
            st.markdown("""
            ### Quick Start
            
            1. **Import the debug logger:**
            ```python
            from debug_logger import debug_logger
            ```
            
            2. **Add logging to your code:**
            ```python
            # Log function calls
            debug_logger.info("Processing PDF file", data={"filename": "example.pdf"})
            
            # Log LLM requests
            debug_logger.llm_request("What is the material?", "gpt-4", 0.7, 1000)
            
            # Log exceptions
            try:
                # your code
                pass
            except Exception as e:
                debug_logger.exception(e)
            
            # Log performance
            with DebugTimer("PDF processing"):
                # your code
                pass
            ```
            
            3. **View logs in this interface**
            
            ### Available Methods
            
            - `debug_logger.info(message, data, context)`
            - `debug_logger.warning(message, data, context)`
            - `debug_logger.error(message, data, context)`
            - `debug_logger.debug(message, data, context)`
            - `debug_logger.llm_request(prompt, model, temp, max_tokens)`
            - `debug_logger.llm_response(model, response, tokens, latency)`
            - `debug_logger.pdf_processing(filename, pages, content)`
            - `debug_logger.web_scraping(url, html, table_data)`
            - `debug_logger.extraction_step(attribute, source, input, output, success)`
            - `debug_logger.exception(exception)`
            - `debug_logger.performance(operation, duration)`
            """)

def create_mini_debug_widget():
    """Create a mini debug widget for embedding in other pages"""
    with st.expander("🔍 Debug Info", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Log Current State"):
                log_streamlit_state()
                st.success("State logged!")
            
            if st.button("📄 View Debug Log"):
                try:
                    with open("debug_log.txt", 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    st.text_area("Debug Log", log_content, height=300)
                except FileNotFoundError:
                    st.info("No debug log file found yet.")
        
        with col2:
            if st.button("🗑️ Clear Log"):
                debug_logger.clear_log()
                st.success("Log cleared!")
            
            # Show current session state
            st.write("**Current Session State:**")
            for key, value in st.session_state.items():
                st.write(f"- {key}: {str(value)[:100]}...")

if __name__ == "__main__":
    create_debug_interface() 