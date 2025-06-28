import streamlit as st
import pandas as pd
import json
from datetime import datetime
from debug_logger import debug_logger, log_streamlit_state
import os

def create_debug_summary():
    """Create a comprehensive debug summary dashboard"""
    st.set_page_config(page_title="Debug Summary", layout="wide")
    
    st.title("📊 Debug Summary Dashboard")
    st.markdown("Real-time overview of extraction process and system state")
    
    # Get debug log content
    log_file = "debug_log.txt"
    log_content = ""
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    
    # Parse log entries
    entries = []
    if log_content:
        lines = log_content.split('\n')
        current_entry = {}
        
        for line in lines:
            if line.startswith('=' * 60):
                if current_entry:
                    entries.append(current_entry)
                    current_entry = {}
                continue
                
            if line.startswith('STEP '):
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
    
    # Create summary statistics
    if entries:
        df = pd.DataFrame(entries)
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Steps", len(entries))
        with col2:
            error_count = len(df[df['level'] == 'ERROR'])
            st.metric("Errors", error_count, delta_color="inverse" if error_count > 0 else "off")
        with col3:
            warning_count = len(df[df['level'] == 'WARNING'])
            st.metric("Warnings", warning_count, delta_color="inverse" if warning_count > 0 else "off")
        with col4:
            info_count = len(df[df['level'] == 'INFO'])
            st.metric("Info Steps", info_count)
        
        # Process breakdown
        st.subheader("🔍 Process Breakdown")
        
        # Extract key process steps
        web_scraping_steps = df[df['message'].str.contains('web scraping', case=False, na=False)]
        stage1_steps = df[df['message'].str.contains('stage1', case=False, na=False)]
        stage2_steps = df[df['message'].str.contains('stage2', case=False, na=False)]
        llm_steps = df[df['message'].str.contains('llm', case=False, na=False)]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Web Scraping Steps", len(web_scraping_steps))
        with col2:
            st.metric("Stage 1 (Web) Steps", len(stage1_steps))
        with col3:
            st.metric("Stage 2 (PDF) Steps", len(stage2_steps))
        with col4:
            st.metric("LLM Calls", len(llm_steps))
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        recent_entries = df.tail(10)
        
        for _, entry in recent_entries.iterrows():
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
        
        # Error analysis
        if error_count > 0:
            st.subheader("❌ Error Analysis")
            error_entries = df[df['level'] == 'ERROR']
            
            for _, entry in error_entries.iterrows():
                with st.expander(f"Error Step {entry['step']} - {entry['message'][:100]}..."):
                    st.error(f"**Error:** {entry['message']}")
                    if entry.get('context'):
                        st.write("**Context:**")
                        st.json(entry['context'])
                    if entry.get('data'):
                        st.write("**Error Details:**")
                        st.json(entry['data'])
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        # Extract timing information
        timing_entries = df[df['message'].str.contains('latency|duration|time', case=False, na=False)]
        if not timing_entries.empty:
            st.write("**Recent Timing Information:**")
            for _, entry in timing_entries.tail(5).iterrows():
                st.write(f"- {entry['message']}")
        
        # Session state overview
        st.subheader("💾 Session State Overview")
        session_keys = list(st.session_state.keys())
        st.write(f"**Total Session Variables:** {len(session_keys)}")
        
        # Show key session variables
        key_vars = ['retriever', 'pdf_chain', 'web_chain', 'processed_files', 'evaluation_results', 
                   'extraction_performed', 'extraction_attempts', 'part_number_input']
        
        for var in key_vars:
            if var in st.session_state:
                value = st.session_state[var]
                if isinstance(value, (list, dict)):
                    st.write(f"**{var}:** {type(value).__name__} with {len(value)} items")
                else:
                    st.write(f"**{var}:** {str(value)[:100]}...")
            else:
                st.write(f"**{var}:** Not set")
        
        # Raw log view
        with st.expander("📄 Raw Debug Log"):
            st.text_area("Complete Debug Log", log_content, height=400)
    
    else:
        st.info("No debug log entries found. Start using the debug logger to see information here.")
        
        # Show current session state
        st.subheader("💾 Current Session State")
        if st.session_state:
            for key, value in st.session_state.items():
                if isinstance(value, (list, dict)):
                    st.write(f"**{key}:** {type(value).__name__} with {len(value)} items")
                else:
                    st.write(f"**{key}:** {str(value)[:100]}...")
        else:
            st.write("No session state variables found.")

if __name__ == "__main__":
    create_debug_summary() 