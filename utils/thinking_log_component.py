import streamlit as st
import time

def thinking_log_component(current_step_text: str, time_elapsed: str, log_content: str, is_active: bool = True):
    """
    Renders a custom "Thinking" component with a log area.

    Args:
        current_step_text (str): The main text to display (e.g., "Thinking", "Exploring bubble UI").
        time_elapsed (str): The time elapsed (e.g., "23s", "1m 15s").
        log_content (str): The smaller text that serves as the log.
        is_active (bool): If True, applies active styling/animation to the icon.
    """

    # --- CSS for the Thinking Log Component ---
    st.markdown("""
        <style>
        .thinking-card-container {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 0.2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            color: #333333;
            border: 1px solid #e0e0e0;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            display: block;
        }
        .thinking-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.8rem;
        }
        .thinking-card-left-section {
            display: flex;
            align-items: center;
            flex-grow: 1;
        }
        .thinking-icon-container {
            width: 30px;
            height: 30px;
            background-color: #6a5acd;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.2em;
            color: white;
            margin-right: 10px;
            flex-shrink: 0;
        }
        .thinking-icon-container.active {
            animation: pulse-thinking 1.5s infinite ease-in-out;
        }
        .thinking-icon-container .icon-content {
            font-weight: bold;
        }
        .thinking-icon-container.active .icon-content {
            animation: spin-thinking 1s linear infinite;
        }
        @keyframes pulse-thinking {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }
        @keyframes spin-thinking {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .thinking-main-text {
            font-weight: 600;
            font-size: 1.1em;
            color: #1e3c72;
            margin-right: 10px;
        }
        .thinking-time-text {
            font-size: 0.9em;
            color: #6c757d;
        }
        .thinking-dropdown-arrow {
            font-size: 1.5em;
            color: #999999;
            cursor: pointer;
            transition: transform 0.2s;
            margin-left: auto;
        }
        .thinking-dropdown-arrow.rotated {
            transform: rotate(180deg);
        }
        .thinking-log-content {
            font-size: 0.85em;
            color: #555555;
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 0.8rem;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
            font-family: monospace;
        }
        .stButton button {
            background-color: #1e3c72 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-size: 0.9em !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Session State for this component (for demo purposes) ---
    # (Removed demo session state logic for importable version)

    # Determine icon class and content
    icon_class = "thinking-icon-container"
    icon_content_html = '<span class="icon-content">⚡</span>'
    if is_active:
        icon_class += " active"

    # Render the HTML for the component
    st.markdown(f"""
        <div class="thinking-card-container">
            <div class="thinking-card-header">
                <div class="thinking-card-left-section">
                    <div class="{icon_class}">
                        {icon_content_html}
                    </div>
                    <span class="thinking-main-text">{current_step_text}</span>
                    <span class="thinking-time-text">{time_elapsed}</span>
                </div>
                <div class="thinking-dropdown-arrow">
                    &#9660; </div>
            </div>
            <div class="thinking-log-content">
                {log_content}
            </div>
        </div>
    """, unsafe_allow_html=True) 