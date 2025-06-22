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

if __name__ == "__main__":
    main()