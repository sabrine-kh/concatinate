# --- Force python to use pysqlite3 based on chromadb docs ---
# This override MUST happen before any other imports that might import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
import streamlit as st

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()



def main():
    st.set_page_config(page_title="LEOPARTS", page_icon="🦁", layout="wide")
    st.markdown(
        """<style>
        [data-testid="stSidebarNav"] {display: none;}
        
        /* Blue band header styling */
        .header-band {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4a90e2 100%);
            color: white;
            padding: 2rem 0;
            margin: -1rem -1rem 2rem -1rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
        }
        
        .header-band h1 {
            font-size: 3.5em;
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
        
        /* Welcome section styling */
        .welcome-section {
            text-align: center;
            padding: 3rem 0;
        }
        
        .welcome-section h2 {
            color: #1e3c72;
            font-size: 2.5em;
            margin-bottom: 1rem;
        }
        
        .welcome-section p {
            color: #2a5298;
            font-size: 1.3em;
            margin-bottom: 2rem;
        }
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
        if st.button("🔍 Debug Interface"):
            st.switch_page("debug_interface.py")
        if st.button("📊 Debug Summary"):
            st.switch_page("debug_summary.py")
    
    # Blue band header with LEONI
    st.markdown("""
        <div class="header-band">
            <h1>LEOPARTS</h1>
            <h2>LEONI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
        <div class="welcome-section">
            <h2>Welcome!</h2>
            <p>Choose a Tool</p>
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

if __name__ == "__main__":
    main()