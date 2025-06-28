import os
import json
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from loguru import logger
import streamlit as st

class DebugLogger:
    """
    Comprehensive debugging logger that captures EVERYTHING in raw detail.
    Logs to both file and console with timestamps and detailed context.
    """
    
    def __init__(self, log_file: str = "debug_log.txt", enable_console: bool = True):
        self.log_file = log_file
        self.enable_console = enable_console
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_counter = 0
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Clear previous log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== DEBUG LOG STARTED: {datetime.now().isoformat()} ===\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write("=" * 80 + "\n\n")
    
    def _log(self, level: str, message: str, data: Optional[Any] = None, context: Optional[Dict] = None):
        """Internal logging method"""
        self.step_counter += 1
        timestamp = datetime.now().isoformat()
        
        # Build log entry
        log_entry = f"\n{'='*60}\n"
        log_entry += f"STEP {self.step_counter} - {level.upper()} - {timestamp}\n"
        log_entry += f"{'='*60}\n"
        log_entry += f"MESSAGE: {message}\n"
        
        if context:
            log_entry += f"CONTEXT: {json.dumps(context, indent=2, default=str)}\n"
        
        if data is not None:
            if isinstance(data, (dict, list)):
                log_entry += f"DATA: {json.dumps(data, indent=2, default=str)}\n"
            else:
                log_entry += f"DATA: {str(data)}\n"
        
        log_entry += f"{'='*60}\n"
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write to debug log: {e}")
        
        # Console output
        if self.enable_console:
            if level == "error":
                logger.error(f"DEBUG [{self.step_counter}]: {message}")
            elif level == "warning":
                logger.warning(f"DEBUG [{self.step_counter}]: {message}")
            elif level == "info":
                logger.info(f"DEBUG [{self.step_counter}]: {message}")
            else:
                logger.debug(f"DEBUG [{self.step_counter}]: {message}")
    
    def info(self, message: str, data: Optional[Any] = None, context: Optional[Dict] = None):
        """Log info level message"""
        self._log("info", message, data, context)
    
    def warning(self, message: str, data: Optional[Any] = None, context: Optional[Dict] = None):
        """Log warning level message"""
        self._log("warning", message, data, context)
    
    def error(self, message: str, data: Optional[Any] = None, context: Optional[Dict] = None):
        """Log error level message"""
        self._log("error", message, data, context)
    
    def debug(self, message: str, data: Optional[Any] = None, context: Optional[Dict] = None):
        """Log debug level message"""
        self._log("debug", message, data, context)
    
    def function_call(self, function_name: str, args: Dict, kwargs: Dict, context: Optional[Dict] = None):
        """Log function call with arguments"""
        self.info(
            f"FUNCTION CALL: {function_name}",
            data={"args": args, "kwargs": kwargs},
            context=context
        )
    
    def function_return(self, function_name: str, return_value: Any, execution_time: float, context: Optional[Dict] = None):
        """Log function return value and execution time"""
        self.info(
            f"FUNCTION RETURN: {function_name}",
            data={"return_value": return_value, "execution_time": execution_time},
            context=context
        )
    
    def llm_request(self, prompt: str, model: str, temperature: float, max_tokens: int, context: Optional[Dict] = None):
        """Log LLM request details"""
        self.info(
            f"LLM REQUEST: {model}",
            data={
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            context=context
        )
    
    def llm_response(self, model: str, response: str, tokens_used: int, latency: float, context: Optional[Dict] = None):
        """Log LLM response details"""
        self.info(
            f"LLM RESPONSE: {model}",
            data={
                "response": response,
                "tokens_used": tokens_used,
                "latency": latency
            },
            context=context
        )
    
    def pdf_processing(self, file_name: str, page_count: int, text_content: str, context: Optional[Dict] = None):
        """Log PDF processing details"""
        self.info(
            f"PDF PROCESSING: {file_name}",
            data={
                "file_name": file_name,
                "page_count": page_count,
                "text_content_preview": text_content[:1000] + "..." if len(text_content) > 1000 else text_content
            },
            context=context
        )
    
    def web_scraping(self, url: str, html_content: str, table_data: Optional[str], context: Optional[Dict] = None):
        """Log web scraping details"""
        self.info(
            f"WEB SCRAPING: {url}",
            data={
                "url": url,
                "html_preview": html_content[:1000] + "..." if len(html_content) > 1000 else html_content,
                "table_data": table_data
            },
            context=context
        )
    
    def extraction_step(self, attribute: str, source: str, input_data: Any, output_data: Any, success: bool, context: Optional[Dict] = None):
        """Log extraction step details"""
        self.info(
            f"EXTRACTION STEP: {attribute} from {source}",
            data={
                "attribute": attribute,
                "source": source,
                "input_data": input_data,
                "output_data": output_data,
                "success": success
            },
            context=context
        )
    
    def session_state(self, state_name: str, state_value: Any, context: Optional[Dict] = None):
        """Log session state changes"""
        self.info(
            f"SESSION STATE: {state_name}",
            data={"state_name": state_name, "state_value": state_value},
            context=context
        )
    
    def exception(self, exception: Exception, context: Optional[Dict] = None):
        """Log exception with full traceback"""
        self.error(
            f"EXCEPTION: {type(exception).__name__}: {str(exception)}",
            data={
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc()
            },
            context=context
        )
    
    def performance(self, operation: str, duration: float, context: Optional[Dict] = None):
        """Log performance metrics"""
        self.info(
            f"PERFORMANCE: {operation}",
            data={"operation": operation, "duration_seconds": duration},
            context=context
        )
    
    def user_action(self, action: str, data: Optional[Any] = None, context: Optional[Dict] = None):
        """Log user actions"""
        self.info(
            f"USER ACTION: {action}",
            data=data,
            context=context
        )
    
    def data_transformation(self, operation: str, input_data: Any, output_data: Any, context: Optional[Dict] = None):
        """Log data transformation steps"""
        self.info(
            f"DATA TRANSFORMATION: {operation}",
            data={
                "operation": operation,
                "input_data": input_data,
                "output_data": output_data
            },
            context=context
        )
    
    def get_log_contents(self) -> str:
        """Get the current contents of the log file"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {e}"
    
    def clear_log(self):
        """Clear the log file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== DEBUG LOG CLEARED: {datetime.now().isoformat()} ===\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write("=" * 80 + "\n\n")

# Global debug logger instance
debug_logger = DebugLogger()

# Decorator for automatic function logging
def debug_log_function(func):
    """Decorator to automatically log function calls and returns"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Log function call
        debug_logger.function_call(
            func.__name__,
            {"args": args},
            kwargs,
            {"function": func.__name__, "module": func.__module__}
        )
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful return
            debug_logger.function_return(
                func.__name__,
                result,
                execution_time,
                {"function": func.__name__, "module": func.__module__}
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log exception
            debug_logger.exception(
                e,
                {"function": func.__name__, "module": func.__module__, "execution_time": execution_time}
            )
            
            raise
    
    return wrapper

# Context manager for timing operations
class DebugTimer:
    """Context manager for timing operations with automatic logging"""
    
    def __init__(self, operation_name: str, context: Optional[Dict] = None):
        self.operation_name = operation_name
        self.context = context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        debug_logger.info(f"TIMER START: {self.operation_name}", context=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        debug_logger.performance(self.operation_name, duration, self.context)
        
        if exc_type:
            debug_logger.exception(exc_val, self.context)

# Utility functions for common debugging scenarios
def log_streamlit_state():
    """Log all current Streamlit session state"""
    for key, value in st.session_state.items():
        debug_logger.session_state(key, value, {"source": "streamlit_session_state"})

def log_dataframe_info(df, name: str):
    """Log DataFrame information"""
    debug_logger.info(
        f"DATAFRAME INFO: {name}",
        data={
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "head": df.head().to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        },
        context={"dataframe_name": name}
    )

def log_json_parsing(original_string: str, parsed_result: Any, success: bool):
    """Log JSON parsing attempts"""
    debug_logger.data_transformation(
        "JSON_PARSING",
        original_string,
        parsed_result,
        {"success": success, "original_length": len(original_string)}
    )

# Export the main logger instance and utilities
__all__ = [
    'debug_logger',
    'DebugLogger',
    'debug_log_function',
    'DebugTimer',
    'log_streamlit_state',
    'log_dataframe_info',
    'log_json_parsing'
] 