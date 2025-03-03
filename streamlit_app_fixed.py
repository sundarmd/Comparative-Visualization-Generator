import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import logging
import traceback
from typing import Optional, List, Dict, Any, Union

# Constants
VISUALIZATION_HEIGHT = 500
MAX_WORKFLOW_HISTORY = 10
D3_VERSION = "https://d3js.org/d3.v7.min.js"

# OpenAI API settings
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.5
MAX_TEMPERATURE = 0.9
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "v1")
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    import openai
    logger.info("Using older OpenAI package")

# Generate a unique key for this session if it doesn't exist
if 'viz_key' not in st.session_state:
    st.session_state.viz_key = int(time.time())
    logger.info(f"Generated session key: {st.session_state.viz_key}")


class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass


def display_loading_animation():
    """Display a loading animation while waiting for the API response."""
    with st.spinner("🎨 Generating visualization..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)


def get_api_key() -> Optional[str]:
    """
    Get the OpenAI API key from environment variables, secrets, or user input.
    
    Returns:
        str: The OpenAI API key or None if not available.
    """
    # Try to get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not in environment, try secrets (for Streamlit Cloud deployment)
    if not api_key and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        
    # If still not available, ask the user
    if not api_key:
        with st.sidebar:
            st.markdown("## OpenAI API Key")
            api_key = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            if api_key:
                st.success("API key provided! 🎉")
                # Test the API key
                if test_api_key(api_key):
                    st.success("✓ API key is valid!")
                else:
                    st.error("✗ API key seems invalid. Please check and try again.")
            else:
                st.warning("Please enter your OpenAI API key to use this app.")
    
    return api_key


def test_api_key(api_key: str) -> bool:
    """
    Test if the provided OpenAI API key is valid.
    
    Args:
        api_key (str): The OpenAI API key to test.
        
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    try:
        if OPENAI_API_VERSION == "v1":
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            return True
        else:
            openai.api_key = api_key
            models = openai.Model.list()
            return True
    except Exception as e:
        logger.error(f"Error testing API key: {str(e)}")
        return False


def preprocess_data(file1, file2) -> pd.DataFrame:
    """
    Preprocess the uploaded CSV files by:
    1. Reading them into DataFrames
    2. Adding a source column to identify each file
    3. Merging them into a single DataFrame
    4. Performing common preprocessing tasks
    
    Args:
        file1: First uploaded CSV file.
        file2: Second uploaded CSV file.
        
    Returns:
        pd.DataFrame: Preprocessed and merged DataFrame.
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Add source columns to identify data origin
        df1['source'] = 'dataset_1'
        df2['source'] = 'dataset_2'
        
        # Merge the DataFrames
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Basic preprocessing
        # 1. Convert to appropriate data types
        numeric_columns = merged_df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            # Ensure numeric columns don't have mixed types
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        
        # 2. Handle date columns
        date_columns = [col for col in merged_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                merged_df[col] = pd.to_datetime(merged_df[col], errors='coerce')
            except:
                pass
        
        # 3. Handle missing values
        merged_df = merged_df.fillna({
            col: 0 if col in numeric_columns else 'Unknown'
            for col in merged_df.columns
        })
        
        # Log preprocessing results
        logger.info(f"Preprocessed data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error preprocessing data: {str(e)}")
        # Return an empty DataFrame if preprocessing fails
        return pd.DataFrame()


def validate_d3_code(code: str) -> bool:
    """
    Validate D3.js code for common issues.
    
    Args:
        code (str): The D3.js code to validate.
        
    Returns:
        bool: True if the code passes validation, False otherwise.
    """
    try:
        # Check for required D3 elements
        if "createVisualization" not in code:
            logger.warning("Missing 'createVisualization' function in D3 code")
            st.warning("⚠️ Missing 'createVisualization' function. The code may not work properly.")
            return False
        
        # Check for access to data parameter
        if not re.search(r'function\s+createVisualization\s*\(\s*data\s*,', code):
            logger.warning("createVisualization function may not properly accept data parameter")
            st.warning("⚠️ The visualization function may not correctly handle data. Check the parameters.")
            
        # Check for potentially unsafe code
        unsafe_patterns = [
            r'document\.write',
            r'eval\(',
            r'setTimeout\(',
            r'setInterval\(',
            r'new\s+Function\(',
            r'fetch\(',
            r'XMLHttpRequest'
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, code):
                logger.warning(f"Potentially unsafe code pattern found: {pattern}")
                st.warning(f"⚠️ Potentially unsafe code pattern detected: {pattern}. This may cause issues.")
        
        # Additional checks for common D3 errors
        if 'd3.select("#viz-svg")' not in code and 'svgElement' not in code:
            logger.warning("Code may not properly use the provided SVG element")
            st.warning("⚠️ The code might not use the provided SVG element. This could cause rendering issues.")
            
        # Check for proper error handling
        if "try" not in code:
            logger.warning("No error handling found in D3 code")
            st.warning("⚠️ No error handling detected in the visualization code. Consider adding try-catch blocks.")
        
        # Check if the code is too short (likely incomplete)
        if len(code) < 50:
            logger.warning("D3 code is suspiciously short")
            st.warning("⚠️ The code is very short and may be incomplete.")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating D3 code: {str(e)}")
        st.error(f"Error validating code: {str(e)}")
        return False


def display_visualization(d3_code: str, placeholder=None) -> None:
    """
    Display a D3.js visualization in Streamlit.
    
    Args:
        d3_code (str): The D3.js code to display.
        placeholder (streamlit.delta_generator.DeltaGenerator, optional): Streamlit placeholder to render the visualization in.
            If None, renders in the current Streamlit position.
    """
    try:
        # Generate a unique timestamp to prevent caching
        timestamp = int(time.time())
        
        # Ensure we have the JSON data available
        if 'json_data' not in st.session_state or st.session_state.json_data is None:
            if 'preprocessed_df' in st.session_state and st.session_state.preprocessed_df is not None:
                st.session_state.json_data = st.session_state.preprocessed_df.to_dict(orient='records')
                logger.info(f"Generated json_data with {len(st.session_state.json_data)} records")
            else:
                logger.warning("No preprocessed data available for visualization")
        
        # Use the visualization container if it exists
        if 'viz_container' not in st.session_state:
            st.session_state.viz_container = placeholder if placeholder else st
        
        # Display the visualization using HTML
        logger.info("Rendering D3 visualization")
        st.session_state.viz_container.components.html(
            html_content=d3_code,
            height=VISUALIZATION_HEIGHT + 50,  # Add some margin
            scrolling=True
        )
        logger.info("Visualization displayed successfully")
    except Exception as e:
        logger.error(f"Error displaying visualization: {str(e)}")
        st.error(f"Error displaying visualization: {str(e)}")
        # Display a fallback message to the user
        if placeholder:
            placeholder.error("Failed to render visualization. Please check the browser console for details.")
        else:
            st.error("Failed to render visualization. Please check the browser console for details.")


def main():
    st.set_page_config(page_title="🎨 Comparative Visualization Generator", page_icon="✨", layout="wide")
    st.title("🎨 Comparative Visualization Generator")

    # Get API key from environment, secrets, or user input
    api_key = get_api_key()
    
    # Initialize session state for workflow history if it doesn't exist
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []  # Stores the history of visualization changes
    
    if 'update_viz' not in st.session_state:
        st.session_state.update_viz = False
    
    # Store container references in session state for stability across reruns
    if 'viz_header' not in st.session_state:
        st.session_state.viz_header = st.empty()
    if 'viz_status' not in st.session_state:
        st.session_state.viz_status = st.empty()
    if 'viz_caption' not in st.session_state:
        st.session_state.viz_caption = st.empty()
    if 'viz_container' not in st.session_state:
        st.session_state.viz_container = st.empty()
    
    # Get container references from session state
    viz_header = st.session_state.viz_header
    viz_status = st.session_state.viz_status
    viz_caption = st.session_state.viz_caption
    viz_container = st.session_state.viz_container
    
    # Display model information in a less prominent place if needed
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")

    st.header("Upload CSV Files")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload first CSV file", type="csv", key="file1_uploader")
    with col2:
        file2 = st.file_uploader("Upload second CSV file", type="csv", key="file2_uploader")

    # Show informational message when no files are uploaded
    if not file1 or not file2:
        st.info("📊 Please upload both CSV files to generate a visualization.")
        st.markdown("""
        ### How to use this app:
        1. Enter your OpenAI API key in the sidebar
        2. Upload two CSV files for comparison
        3. The app will generate an initial visualization
        4. Describe changes you want in the text field
        5. Click 'Update Visualization' to apply your requests
        """)
        
        # Clear any existing visualizations if files are removed
        if ('current_viz' in st.session_state and st.session_state.current_viz is not None):
            # Clear visualization containers
            viz_header.empty()
            viz_status.empty()
            viz_caption.empty()
            with viz_container.container():
                st.empty()
            
            # Reset session state
            st.session_state.current_viz = None
            st.session_state.preprocessed_df = None
            st.session_state.json_data = None
            st.session_state.history_index = 0
        
        return  # Exit early if files not uploaded

    # More code would follow for the rest of the application
    # ...

if __name__ == "__main__":
    main()
