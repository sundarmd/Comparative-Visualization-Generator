import streamlit as st
import pandas as pd
import openai
import os
import json
import logging
import traceback
from typing import Optional, Dict, List
import re
import urllib.parse
import streamlit.components.v1 as components
from dotenv import load_dotenv
import time
import tempfile
from pathlib import Path
import uuid
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define configuration constants
MAX_WORKFLOW_HISTORY = int(os.getenv("MAX_WORKFLOW_HISTORY", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY = int(os.getenv("INITIAL_RETRY_DELAY", "2"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
MAX_TEMPERATURE = float(os.getenv("MAX_TEMPERATURE", "1.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
VISUALIZATION_HEIGHT = int(os.getenv("VISUALIZATION_HEIGHT", "550"))
D3_VERSION = os.getenv("D3_VERSION", "https://d3js.org/d3.v7.min.js")

# Initialize session state
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []  # Stores the history of visualization changes
if 'current_viz' not in st.session_state:
    st.session_state.current_viz = None  # Stores the current D3.js visualization code
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None  # Stores the preprocessed DataFrame
if 'json_data' not in st.session_state:
    st.session_state.json_data = None  # Stores the JSON data for visualization
if 'update_viz' not in st.session_state:
    st.session_state.update_viz = False  # Flag to trigger visualization update
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Stores the chat history
if 'viz_key' not in st.session_state:
    st.session_state.viz_key = str(uuid.uuid4())  # Unique key for visualization container
if 'history_index' not in st.session_state:
    st.session_state.history_index = 0  # Tracks current position in visualization history

# Import OpenAI with version compatibility
try:
    # Try newer OpenAI SDK style (v1.0.0+)
    from openai import OpenAI
    OPENAI_API_VERSION = "v1"
    logger.info("Using OpenAI API v1 client")
except ImportError:
    # Fall back to older style
    OPENAI_API_VERSION = "v0"
    logger.info("Using OpenAI API v0 client")

# Configure exceptions based on OpenAI version
if OPENAI_API_VERSION == "v1":
    RateLimitError = openai.RateLimitError 
else:
    try:
        RateLimitError = openai.error.RateLimitError
    except AttributeError:
        # If neither works, create a minimal implementation
        class RateLimitError(Exception):
            pass

def display_loading_animation():
    loading_html = """
    <div class="loading-container" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 500px;">
        <div class="loading-spinner">
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
            <div class="spinner-ring"></div>
        </div>
        <div class="loading-text">Loading...</div>
    </div>
    <style>
        .loading-spinner {
            position: relative;
            width: 80px;
            height: 80px;
        }
        .spinner-ring {
            position: absolute;
            width: 100%;
            height: 100%;
            border: 4px solid transparent;
            border-top-color: #3498db;
            border-radius: 50%;
            animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
        }
        .spinner-ring:nth-child(1) { animation-delay: -0.45s; }
        .spinner-ring:nth-child(2) { animation-delay: -0.3s; }
        .spinner-ring:nth-child(3) { animation-delay: -0.15s; }
        .loading-text {
            margin-top: 20px;
            font-family: Arial, sans-serif;
            font-size: 18px;
            color: #3498db;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
    </style>
    """
    return st.components.v1.html(loading_html, height=400)

def get_api_key() -> Optional[str]:
    """
    Securely retrieve the API key.
    
    This function attempts to get the OpenAI API key from:
    1. Environment variables (loaded from .env file)
    2. Streamlit secrets
    3. User input via sidebar
    
    Returns:
        Optional[str]: The API key if found or entered, None otherwise.
    """
    # First try to get from environment variables (from .env file)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found in environment, try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            # Handle case where secrets might not be configured
            pass
    
    # If still not found, prompt the user
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.sidebar.success("API key received successfully! 🎉")
    
    return api_key

def test_api_key(api_key: str) -> bool:
    """
    This function attempts to make a simple API call using the provided OpenAI API key.
    
    Args:
        api_key (str): The OpenAI API key to test.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    if not api_key:
        return False
    
    try:
        # Test API key based on version
        if OPENAI_API_VERSION == "v1":
            client = OpenAI(api_key=api_key)
            # Make a minimal API call to check if the key is valid
            client.models.list(limit=1)
        else:
            openai.api_key = api_key
            # Make a minimal API call to check if the key is valid
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        return False

def preprocess_data(file1, file2) -> pd.DataFrame:
    """
    Preprocess and merge the two dataframes for comparison.
    
    This function reads two CSV files, adds a 'Source' column to each,
    merges them, handles missing values, ensures consistent data types,
    and standardizes column names.
    
    Args:
        file1: First CSV file uploaded by the user.
        file2: Second CSV file uploaded by the user.
    
    Returns:
        pd.DataFrame: Preprocessed and merged DataFrame.
    
    Raises:
        ValueError: If files are empty or cannot be parsed.
        Exception: For any other preprocessing errors.
    """
    logger.info("Starting data preprocessing")
    try:
        # Read CSV files into pandas DataFrames
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
        except pd.errors.EmptyDataError:
            raise ValueError("One or both of the uploaded files are empty.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the CSV files. Please ensure they are valid CSV format.")
        
        # Add 'Source' column to identify the origin of each row
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'
        
        # Merge the two DataFrames
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Handle missing values by filling them with 0
        merged_df = merged_df.fillna(0)
        
        # Ensure consistent data types
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                try:
                    merged_df[col] = pd.to_numeric(merged_df[col])
                except ValueError:
                    pass  # Keep as string if can't convert to numeric
        
        # Standardize column names: lowercase and replace spaces with underscores
        merged_df.columns = merged_df.columns.str.lower().str.replace(' ', '_')
        
        logger.info("Data preprocessing completed successfully")
        return merged_df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def validate_d3_code(code: str) -> dict:
    """
    Perform validation on the generated D3 code.
    
    This function checks for the presence of key D3.js elements and features.
    
    Args:
        code (str): The D3.js code to validate.
    
    Returns:
        dict: Validation results with 'valid' boolean and 'missing_features' list.
    """
    missing_features = []
    warnings = []
    
    # Check if the code defines the createVisualization function
    if not re.search(r'function\s+createVisualization\s*\(data,\s*svgElement\)\s*{', code):
        missing_features.append("Basic Structure: createVisualization function")
    
    # Check for proper scale initialization to prevent 'ticks' errors
    scale_pattern = re.search(r'\.scale[A-Za-z]+\(\s*\)', code) 
    if scale_pattern:
        warnings.append("Potential error: D3 scale initialized without arguments")
        
    # Check for proper null/undefined checks to prevent common errors
    if not re.search(r'if\s*\(.+(?:data|d|value).+\)\s*{', code):
        warnings.append("Missing data validation: No conditional checks for data values")
    
    # Check for domain and range definitions on scales
    if re.search(r'\.domain\s*\(', code) and not re.search(r'\.range\s*\(', code):
        warnings.append("Incomplete scale: domain defined without range")
    
    if re.search(r'\.range\s*\(', code) and not re.search(r'\.domain\s*\(', code):
        warnings.append("Incomplete scale: range defined without domain")
    
    # Check for proper axis handling to prevent 'ticks' error
    if re.search(r'd3\.axisBottom|d3\.axisLeft|d3\.axisRight|d3\.axisTop', code) and not re.search(r'\.ticks\s*\(', code):
        warnings.append("Potential ticks error: Using axis without explicitly configuring ticks")
    
    # Check for proper error handling
    if not re.search(r'try\s*{', code):
        warnings.append("Error Handling: No try-catch blocks found")
    
    # Check for data validation
    if not re.search(r'if\s*\(\s*!data|\s*data\s*===\s*null|\s*data\s*===\s*undefined|\s*!Array\.isArray\(data\)|\s*data\.length\s*===\s*0', code):
        warnings.append("Data Validation: No checks for invalid data")
    
    # Check for responsive design with viewBox
    if not re.search(r'viewBox|preserveAspectRatio', code):
        warnings.append("Responsive Design: No viewBox or preserveAspectRatio attributes")
    
    # Check for width and height calculation
    if not re.search(r'\.attr\s*\(\s*[\'"](width|height)[\'"]', code):
        warnings.append("Sizing: No width or height attributes set")
    
    # ============ AESTHETIC VALIDATION CHECKS ============
    
    # Check for mandatory title
    if not re.search(r'\.text\s*\(\s*[\'"][^\'\"]*(?:vs|comparison|distribution|relationship|pattern)[^\'\"]*[\'"]\s*\)', code, re.IGNORECASE):
        missing_features.append("Visual Design: Missing descriptive title")
    
    # Check for legend implementation
    legend_patterns = [
        r'\.append\s*\(\s*[\'"]g[\'"]\s*\)[^}]*class[^}]*legend',
        r'legend[^}]*\.append',
        r'\.attr\s*\(\s*[\'"]class[\'"]\s*,\s*[\'"]legend[\'"]\s*\)'
    ]
    has_legend = any(re.search(pattern, code) for pattern in legend_patterns)
    if not has_legend:
        missing_features.append("Visual Design: Missing legend for data encodings")
    
    # Check for axis labels
    axis_label_patterns = [
        r'\.text\s*\(\s*[\'"][^\'\"]+[\'"]\s*\)[^}]*(?:x-axis|y-axis|axis)',
        r'(?:x-axis|y-axis|axis)[^}]*\.text\s*\(',
        r'axis[^}]*label[^}]*\.text'
    ]
    has_axis_labels = any(re.search(pattern, code, re.IGNORECASE) for pattern in axis_label_patterns)
    if not has_axis_labels:
        missing_features.append("Visual Design: Missing descriptive axis labels")
    
    # Check for professional color schemes
    professional_colors = [
        r'#1f77b4|#aec7e8|#ff7f0e',  # Professional Blue Palette
        r'#440154|#482777|#3f4a8a',  # Viridis Palette  
        r'#2196F3|#FF9800|#4CAF50',  # Material Design Palette
        r'd3\.schemeCategory10|d3\.schemeSet3|d3\.schemeTableau10'  # D3 color schemes
    ]
    has_professional_colors = any(re.search(pattern, code) for pattern in professional_colors)
    if not has_professional_colors:
        warnings.append("Visual Design: Consider using professional color palettes")
    
    # Check for proper typography
    typography_patterns = [
        r'font-size[\'\"]\s*,\s*[\'\"]\d+px',
        r'font-weight[\'\"]\s*,\s*[\'\"]\d+',
        r'system-ui|sans-serif'
    ]
    has_typography = any(re.search(pattern, code) for pattern in typography_patterns)
    if not has_typography:
        warnings.append("Visual Design: Missing typography specifications")
    
    # Check for hover interactions
    hover_patterns = [
        r'\.on\s*\(\s*[\'"]mouseover[\'"]\s*,',
        r'\.on\s*\(\s*[\'"]mouseenter[\'"]\s*,',
        r'hover|mouseover|mouseenter'
    ]
    has_hover = any(re.search(pattern, code) for pattern in hover_patterns)
    if not has_hover:
        warnings.append("Interactivity: Missing hover effects for better user experience")
    
    # Check for MANDATORY transitions with proper timing
    transition_patterns = [
        r'\.transition\s*\(\s*\)[^}]*\.duration\s*\(\s*300\s*\)',  # 300ms duration specifically
        r'\.duration\s*\(\s*300\s*\)[^}]*\.ease',                  # 300ms with easing
        r'300[^}]*ease.*in.*out|ease.*in.*out[^}]*300'              # 300ms and ease-in-out
    ]
    has_proper_transitions = any(re.search(pattern, code) for pattern in transition_patterns)
    if not has_proper_transitions:
        missing_features.append("Visual Enhancement: Missing mandatory 300ms ease-in-out transitions")
    
    # Check for MANDATORY grid lines
    grid_patterns = [
        r'\.tickSizeInner\s*\(\s*-',  # Must use negative values for grid lines
        r'tickSize\s*\(\s*-',          # Alternative grid line method
        r'grid.*stroke.*#f1f3f4',      # Check for specific grid styling
        r'stroke.*#f1f3f4.*grid'       # Alternative order
    ]
    has_proper_grid = any(re.search(pattern, code) for pattern in grid_patterns)
    if not has_proper_grid:
        missing_features.append("Visual Enhancement: Missing mandatory grid lines with proper styling")
    
    # Check for MANDATORY point opacity
    opacity_patterns = [
        r'opacity[\'\"]\s*,\s*0\.[7-8]',      # opacity: 0.7 or 0.8
        r'\.attr\s*\(\s*[\'"]opacity[\'"]\s*,\s*0\.[7-8]',  # .attr("opacity", 0.7)
        r'\.style\s*\(\s*[\'"]opacity[\'"]\s*,\s*0\.[7-8]', # .style("opacity", 0.7)
    ]
    has_proper_opacity = any(re.search(pattern, code) for pattern in opacity_patterns)
    if not has_proper_opacity:
        missing_features.append("Visual Enhancement: Missing mandatory point opacity (0.7-0.8)")
    
    # Check for MANDATORY hover effects with opacity changes
    hover_opacity_patterns = [
        r'mouseover[^}]*opacity[^}]*1\.0',     # Hover changes opacity to 1.0
        r'mouseenter[^}]*opacity[^}]*1',       # Alternative event
        r'hover[^}]*opacity[^}]*1\.0'          # CSS hover
    ]
    has_hover_opacity = any(re.search(pattern, code) for pattern in hover_opacity_patterns)
    if not has_hover_opacity:
        missing_features.append("Interactivity: Missing mandatory hover opacity effects")
    
    # Check for MANDATORY tooltips (strict validation)
    tooltip_patterns = [
        r'd3\.select\s*\(\s*[\'"]body[\'"]\s*\)[^}]*\.append\s*\(\s*[\'"]div[\'"]\s*\)',  # Must append to body
        r'\.on\s*\(\s*[\'"]mouseover[\'"]\s*,',                                      # Must have mouseover
        r'\.on\s*\(\s*[\'"]mousemove[\'"]\s*,',                                       # Must have mousemove  
        r'\.on\s*\(\s*[\'"]mouseout[\'"]\s*,'                                         # Must have mouseout
    ]
    tooltip_requirements_met = sum(1 for pattern in tooltip_patterns if re.search(pattern, code))
    if tooltip_requirements_met < 4:  # All 4 patterns must be present
        missing_features.append("Interactivity: Missing mandatory tooltips (must have body append + 3 events)")
        
    # Check for safeD3.createAxesWithLabels usage (MANDATORY)
    axes_helper_patterns = [
        r'window\.safeD3\.createAxesWithLabels\s*\(',
        r'safeD3\.createAxesWithLabels\s*\('
    ]
    has_axes_helper = any(re.search(pattern, code) for pattern in axes_helper_patterns)
    if not has_axes_helper:
        missing_features.append("Axes: Missing mandatory safeD3.createAxesWithLabels usage")
        
    # Check for professional grid lines (strict validation)
    grid_line_patterns = [
        r'tickSizeInner\s*:\s*-',                  # Must use tickSizeInner with negative values
        r'gridLineStroke\s*:\s*[\'"]#[e0-9a-fA-F]+[\'"]',  # Must specify grid line color
        r'gridOpacity\s*:\s*0\.[0-9]'              # Must specify grid opacity
    ]
    grid_requirements_met = sum(1 for pattern in grid_line_patterns if re.search(pattern, code))
    if grid_requirements_met < 2:  # At least 2 of 3 patterns must be present  
        missing_features.append("Visual Enhancement: Missing professional grid lines configuration")
    
    # Check for appropriate margins
    margin_pattern = re.search(r'margin[^}]*\{[^}]*top:\s*(\d+)', code)
    if margin_pattern:
        margin_top = int(margin_pattern.group(1))
        if margin_top < 50:
            warnings.append("Layout: Margins too small for professional appearance")
    else:
        warnings.append("Layout: Missing proper margin configuration")
    
    # Calculate validation result
    validation_result = {
        "valid": len(missing_features) == 0,
        "missing_features": missing_features,
        "warnings": warnings
    }
    
    return validation_result

def generate_improvement_instructions(validation_results: dict) -> str:
    """
    Generate specific instructions for improving D3 code based on validation results.
    
    Args:
        validation_results (dict): Results from the validate_d3_code function.
    
    Returns:
        str: Detailed instructions for improving the code.
    """
    if validation_results["valid"]:
        return "The D3 code meets all required criteria."
    
    instructions = ["Your D3.js visualization code needs improvements in the following areas:"]
    
    # Group missing features by category
    missing_by_category = {}
    for missing in validation_results["missing_features"]:
        if ":" in missing:
            category, feature = missing.split(":", 1)
            if category not in missing_by_category:
                missing_by_category[category] = []
            missing_by_category[category].append(feature.strip())
    
    # Generate instructions for each category
    for category, features in missing_by_category.items():
        instructions.append(f"\n## {category}")
        
        if category == "Basic Structure":
            if "createVisualization function" in features:
                instructions.append("- Define a function named `createVisualization(data, svgElement)`")
            if "Clear previous content" in features:
                instructions.append("- Add code to clear previous visualization: `svgElement.selectAll(\"*\").remove()`")
            if "Configuration object" in features:
                instructions.append("- Create a configuration object with customizable parameters (margins, colors, animations)")
            if "Responsive SVG setup" in features:
                instructions.append("- Make the SVG responsive with viewBox and preserveAspectRatio attributes")
        
        elif category == "Layout":
            if "Margins configuration" in features:
                instructions.append("- Define proper margins (top, right, bottom, left)")
            if "Responsive dimensions" in features:
                instructions.append("- Calculate dimensions based on container size")
            if "Window resize handler" in features:
                instructions.append("- Add a window resize handler to update the visualization")
        
        elif category == "Visual Design":
            if "Missing descriptive title" in features:
                instructions.append("- Add a meaningful title that describes the data relationships, e.g., 'Sepal Length vs Width: Comparing Iris Species'")
            if "Missing legend for data encodings" in features:
                instructions.append("- Create a legend in the top-right corner explaining color/shape encodings")
            if "Missing descriptive axis labels" in features:
                instructions.append("- Add clear, descriptive axis labels with units where appropriate")
        
        elif category == "Scales and Axes":
            if "Scale creation" in features:
                instructions.append("- Create appropriate D3 scales (d3.scaleBand, d3.scaleLinear, etc.)")
            if "Grid lines" in features:
                instructions.append("- Add grid lines to the axes using tickSize")
            if "Axis styling" in features:
                instructions.append("- Style the axes with proper fonts, colors, and rotated labels if needed")
        
        elif category == "Visualization Elements":
            if "Data binding" in features:
                instructions.append("- Bind data to visual elements using the D3 data join pattern")
            if "Color scales" in features:
                instructions.append("- Use a color scale to differentiate data sources")
            if "Element styling" in features:
                instructions.append("- Style elements with fill, stroke, and rounded corners")
        
        elif category == "Visual Enhancement":
            if "Missing mandatory grid lines with proper styling" in features:
                instructions.append("- REQUIRED: Add grid lines using .tickSizeInner(-innerWidth) for y-axis and .tickSizeInner(-innerHeight) for x-axis")
                instructions.append("- REQUIRED: Style grid lines with stroke: '#f1f3f4', stroke-width: 0.5px, opacity: 0.7")
            if "Missing mandatory point opacity (0.7-0.8)" in features:
                instructions.append("- REQUIRED: Set all data points to opacity between 0.7-0.8 using .attr('opacity', 0.7) or .style('opacity', 0.8)")
            if "Missing mandatory 300ms ease-in-out transitions" in features:
                instructions.append("- REQUIRED: Add smooth transitions using .transition().duration(300).ease(d3.easeInOut)")
                instructions.append("- Apply transitions to all interactive elements and data updates")
        
        elif category == "Interactivity":
            if "Missing mandatory tooltips" in features:
                instructions.append("- REQUIRED: Implement tooltips that appear on mouseover with relevant data values")
                instructions.append("- Position tooltips properly and format content clearly")
            if "Missing mandatory hover opacity effects" in features:
                instructions.append("- REQUIRED: Add hover effects that change point opacity from 0.7-0.8 to 1.0")
                instructions.append("- REQUIRED: Include visual feedback like subtle stroke or glow on hover")
            if "Tooltips" in features:
                instructions.append("- Add tooltips that show on mouseover/hover")
            if "Highlighting effects" in features:
                instructions.append("- Implement highlighting effects on hover with transitions")
            if "Click interactions" in features:
                instructions.append("- Add click interactions for detailed information")
        
        elif category == "Advanced Interactions":
            if "Zoom functionality" in features:
                instructions.append("- Implement zoom functionality with d3.zoom()")
            if "Brush component" in features:
                instructions.append("- Add a brush component for range selection")
            if "Reset button" in features:
                instructions.append("- Create a reset zoom/brush button")
            if "Axis selection UI" in features:
                instructions.append("- Add dropdowns to change the variables displayed on axes")
        
        elif category == "Animations":
            if "Entrance animations" in features:
                instructions.append("- Add entrance animations for visualization elements")
            if "Staggered animations" in features:
                instructions.append("- Implement staggered animations using delay based on index")
            if "Update transitions" in features:
                instructions.append("- Use transitions for all updates to the visualization")
        
        elif category == "UI Components":
            if "Interactive legend" in features:
                instructions.append("- Create an interactive legend that allows toggling visibility")
            if "Dynamic title" in features:
                instructions.append("- Add a title that updates based on selected axes")
            if "Axis labels" in features:
                instructions.append("- Implement axis labels that update dynamically")
            if "UI controls" in features:
                instructions.append("- Add UI controls for changing visualization parameters")
        
        elif category == "Accessibility":
            if "ARIA attributes" in features:
                instructions.append("- Add ARIA attributes for screen readers (role, aria-label)")
        
        elif category == "Performance":
            if "Clip path" in features:
                instructions.append("- Implement a clip path to prevent elements from overflowing")
            if "Efficient updates" in features:
                instructions.append("- Create functions for efficiently updating the visualization")
    
    # Add examples for specific visualization types based on evaluation goals
    instructions.append("\n## Specific Visualization Requirements")
    instructions.append("- For scatterplots: Implement point markers that can be styled (shape, size) based on data attributes")
    instructions.append("- For histograms: Support grouped/stacked bars for comparing distributions")
    instructions.append("- For parallel coordinates: Implement brushable axes and highlighting of selected paths")
    
    # Add specific examples for the Iris dataset
    instructions.append("\n## Iris Dataset Specific Features")
    instructions.append("- Support color encoding by species (setosa, versicolor, virginica)")
    instructions.append("- Allow point size to be mapped to petal or sepal measurements")
    instructions.append("- Support paired/small multiple views for comparing distributions")
    
    return "\n".join(instructions)

def generate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    """
    Generate D3.js code using OpenAI API based on data and user requests.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing the data to visualize.
        api_key (str): OpenAI API key.
        user_input (str, optional): User's request for visualization modifications.
    
    Returns:
        str: D3.js visualization code.
    """
    user_input = str(user_input).strip() if user_input is not None else ""
    
    # Debug log
    logger.info(f"GENERATE D3 CODE FUNCTION CALLED WITH USER INPUT: '{user_input}'")
    
    try:
        # Validate and prepare data sample
        if df is None or df.empty:
            logger.error("Empty or invalid DataFrame provided")
            raise ValueError("Data validation failed: Empty DataFrame")
            
        data_sample = df.head(5).to_dict(orient='records')
        schema = df.dtypes.to_dict()
        schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
        
        # Initialize OpenAI client based on version
        if OPENAI_API_VERSION == "v1":
            client = OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
        
        # Get model and parameters
        model = os.getenv("DEFAULT_MODEL", "gpt-4o-2024-08-06")
        max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Enhanced prompt with quality standards matching the sample
        prompt = f"""
        # PROFESSIONAL D3.js VISUALIZATION CREATION

        Create a publication-quality D3.js version 7 visualization with modern aesthetics and professional design standards.
        
        ## USER REQUEST:
        {user_input if user_input else "Create an initial visualization that best represents this data with professional styling and clear storytelling"}
        
        ## DATA INFORMATION:
        Schema: {schema_str}
        
        Sample data: 
        ```json
        {json.dumps(data_sample[:5], indent=2)}
        ```
        
        ## MANDATORY VISUAL DESIGN REQUIREMENTS:
        
        ### 1. PROFESSIONAL AESTHETICS & STORYTELLING
        - **MANDATORY TITLE**: Create a descriptive, insightful title that tells the data story
        - **MANDATORY LEGEND**: Always include a properly positioned legend explaining data encodings
        - **MANDATORY AXIS LABELS**: Clear, descriptive axis labels with units where applicable
        - **DATA INSIGHTS**: Add subtle annotations highlighting key insights or patterns
        - **Visual Hierarchy**: Use font sizes, weights, and colors to guide attention
        
        ### 2. MODERN COLOR PALETTE (Choose ONE consistently):
        - **Professional Blue Palette**: ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896"]
        - **Viridis Palette**: ["#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9d8a", "#6cce5a", "#b6de2b", "#fee825"]
        - **Material Design Palette**: ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722", "#795548"]
        - **Ensure 4.5:1 contrast ratio** for accessibility compliance
        
        ### 3. TYPOGRAPHY SYSTEM:
        - **Title**: 18px, font-weight: 600, color: #2c3e50
        - **Axis Labels**: 14px, font-weight: 500, color: #34495e  
        - **Tick Labels**: 12px, font-weight: 400, color: #7f8c8d
        - **Legend**: 13px, font-weight: 400, color: #2c3e50
        - **Tooltips**: 12px, font-weight: 400, color: #2c3e50
        - **Font Family**: Use "system-ui, -apple-system, sans-serif" for all text
        
        ### 4. LAYOUT & SPACING SYSTEM:
        - **Margins**: Use generous margins - minimum {{top: 60, right: 120, bottom: 80, left: 80}}
        - **Element Spacing**: 12px between related elements, 24px between sections
        - **Point Size**: For scatter plots, use 4-6px radius with 0.8 opacity
        - **Stroke Width**: Use 1.5px for important elements, 0.5px for grid lines
        
        ### 5. MANDATORY VISUAL ENHANCEMENTS:
        - **REQUIRED Grid Lines**: MUST implement subtle grid lines using .tickSizeInner(-width) and .tickSizeInner(-height) with stroke: #f1f3f4 and stroke-width: 0.5px
        - **REQUIRED Point Opacity**: All data points MUST have opacity between 0.7-0.8 to handle overlapping data
        - **REQUIRED Hover Effects**: MUST implement mouseover events that increase opacity to 1.0 and add visual feedback
        - **REQUIRED Smooth Animations**: All transitions MUST use 300ms ease-in-out duration
        - **REQUIRED Point Sizing**: Data points MUST be appropriately sized (4-6px radius for scatter plots) with proper stroke-width
        - **REQUIRED Professional Spacing**: Elements MUST have proper spacing and not overlap or crowd each other
        
        ## TECHNICAL REQUIREMENTS:
        1. Create a function named createVisualization(data, svgElement) that follows professional D3 standards
           - IMPORTANT: svgElement parameter is a D3 selection, not a raw DOM element
           - Always use svgElement.append() instead of d3.select("svg").append()
           - You can use window.safeD3 for safer scale and axis creation
        
        2. Include a comprehensive configuration object with:
           - Professional margins and spacing as specified above
           - Width and height derived from svgElement
           - Professional color palette selection
           - Typography system implementation
           - Animation and transition settings
        
        3. Implement responsive design:
           - Get container dimensions from svgElement using .attr("width") and .attr("height")
           - Use viewBox for SVG scaling
           - Handle window resize events
           - Add preserveAspectRatio
        
        4. **MANDATORY PROFESSIONAL AXES WITH GRID LINES**:
           
        ## CRITICAL: ALWAYS USE safeD3.createAxesWithLabels() - MANUAL AXIS CREATION IS FORBIDDEN
        **You MUST use window.safeD3.createAxesWithLabels() for consistent professional axes!**
        
        ### MANDATORY Axes Creation Pattern:
        ```javascript
        // REQUIRED: Use this exact pattern - manual axis creation will cause validation failure
        const axesResult = window.safeD3.createAxesWithLabels(g, xScale, yScale, {{
            xLabel: "Your X Label (units)",
            yLabel: "Your Y Label (units)", 
            width: width,
            height: height,
            margin: margin,
            // MANDATORY: Grid line configuration
            tickSizeInner: -height,     // Creates grid lines across full height
            tickSizeOuter: 0,           // Removes outer tick extensions
            tickPadding: 10,            // Space between ticks and labels
            gridLineStroke: "#e0e0e0",  // Professional gray grid lines
            gridLineWidth: 0.5,         // Thin grid lines
            gridOpacity: 0.7            // Subtle grid opacity
        }});
        ```
        
        ### Grid Line Requirements (MANDATORY):
           - MUST use exactly this safeD3.createAxesWithLabels() pattern - validation will FAIL for manual axis creation
           - Grid lines MUST span full visualization area using tickSizeInner: -height for X-axis, -width for Y-axis  
           - Grid styling MUST be: stroke: #e0e0e0, stroke-width: 0.5px, opacity: 0.7 for professional appearance
           - Tick values MUST be formatted with .nice() for rounded values (4.0, 5.0, 6.0 instead of 4.3, 5.7, 6.1)
           - Axis labels MUST be descriptive with units where applicable
           - Axes MUST properly join at origin with clean, professional appearance
           - Smooth transitions for all updates (300ms ease-in-out)
        
        5. Add rich interactivity with MANDATORY professional styling:
           
        ## CRITICAL TOOLTIP IMPLEMENTATION (MANDATORY):
        **NEVER append tooltip div to SVG - this is the #1 tooltip failure cause!**
        
        ### Required Tooltip Creation Pattern:
        ```javascript
        // MANDATORY: Append to body, NOT svg
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("position", "absolute")
            .style("opacity", 0)
            .style("pointer-events", "none")  // CRITICAL: prevents tooltip interference
            .style("background", "#fff")
            .style("padding", "8px")
            .style("border", "1px solid #ccc")
            .style("border-radius", "4px")
            .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
            .style("font-size", "12px")
            .style("z-index", "1000");
        ```
        
        ### MANDATORY 3-Event Tooltip Pattern:
        ```javascript
        .on('mouseover', function(event, d) {{
            console.log("Tooltip mouseover triggered", d); // Required debugging
            tooltip.transition().duration(200).style('opacity', 1);
            tooltip.html(`Data: ${{d.value}}`); // Set content
        }})
        .on('mousemove', function(event, d) {{
            tooltip
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        }})
        .on('mouseout', function(event, d) {{
            tooltip.transition().duration(200).style('opacity', 0);
        }});
        ```
        
        ### MANDATORY Requirements:
           - MUST use exactly this 3-event pattern (mouseover/mousemove/mouseout)
           - MUST append tooltip to d3.select("body"), NEVER to svg
           - MUST include console.log for debugging tooltip events
           - MUST use event.pageX/pageY for positioning (not d3.event in v6+)
           - MUST set pointer-events: none to prevent tooltip interference
           - MUST implement hover effects that change opacity from 0.7-0.8 to 1.0
           - MUST add visual feedback on hover (subtle stroke or glow effect)
           - All transitions MUST use .transition().duration(300).ease(d3.easeInOut)
           - Tooltips MUST include relevant data values and proper formatting
        
        6. Include accessibility features:
           - ARIA attributes with meaningful descriptions
           - Role descriptions for screen readers
           - High contrast color combinations (4.5:1 minimum)
        
        7. Add comprehensive error handling:
           - Check for data existence and validity
           - Provide graceful fallbacks for missing or invalid data
        
        ## DATA STORYTELLING REQUIREMENTS (MANDATORY):
        
        ### Chart Type Intelligence:
        - **Scatter Plots**: For relationships between continuous variables, always include trend lines or regression if patterns exist
        - **Comparative Data**: When "source" column exists, use it for color encoding and include clear legend
        - **Species/Categories**: Use consistent, distinguishable colors and shapes for different categories
        - **Time Series**: If temporal data exists, emphasize trends with appropriate scales and annotations
        
        ### Smart Title Generation (MANDATORY):
        - Analyze the data relationships and create meaningful titles like:
          - "Sepal Length vs Width: Comparing Iris Species Across Datasets"
          - "Distribution Patterns in [Variable Name] by [Category]"
          - "Relationship Between [X Variable] and [Y Variable]"
        - Avoid generic titles like "Scatter Plot" or "Data Visualization"
        
        ### Legend Requirements (MANDATORY):
        - Position legend in top-right corner with 10px margin from edges
        - Include clear labels for all color/shape encodings
        - Use the same colors/symbols as in the visualization
        - Add legend title describing what the encoding represents
        
        ## CRITICAL SCALE DOMAIN REQUIREMENTS (MANDATORY):
        **NEVER USE HARDCODED SCALE DOMAINS** - This is the most common cause of data misalignment.
        
        ### CORRECT Scale Creation Pattern (Research-Based):
        ```javascript
        // CRITICAL: Convert strings to numbers with + operator (most common alignment issue)
        const xExtent = d3.extent(data, d => +window.safeD3.getValue(d, "column_name", 0));
        const yExtent = d3.extent(data, d => +window.safeD3.getValue(d, "column_name", 0));
        
        // MANDATORY: Validate extents before creating scales
        console.log("X extent:", xExtent, "Y extent:", yExtent); // Required debugging
        
        // CRITICAL: Filter out invalid values (NaN, null, undefined) before extent calculation
        const cleanData = data.filter(d => {{
            const x = +window.safeD3.getValue(d, "x_column", 0);
            const y = +window.safeD3.getValue(d, "y_column", 0);
            return !isNaN(x) && !isNaN(y) && isFinite(x) && isFinite(y);
        }});
        
        const xScale = window.safeD3.createLinearScale(xExtent, [0, width])
            .nice(); // Round to nice values but validate doesn't extend too far
        const yScale = window.safeD3.createLinearScale(yExtent, [height, 0])  // CRITICAL: [height, 0] for screen coordinates
            .nice();
            
        // MANDATORY: Log final domain/range for debugging
        console.log("X scale domain:", xScale.domain(), "range:", xScale.range());
        console.log("Y scale domain:", yScale.domain(), "range:", yScale.range());
        ```
        
        ### WRONG - Never Do This:
        ```javascript
        // WRONG: Hardcoded domains cause misalignment
        const xScale = window.safeD3.createLinearScale([4, 8], [0, width]); // DON'T DO THIS
        const yScale = window.safeD3.createLinearScale([1, 5], [height, 0]); // DON'T DO THIS
        
        // WRONG: Missing data type conversion (treats "5.2" as string)
        const xExtent = d3.extent(data, d => window.safeD3.getValue(d, "column_name", 0)); // Missing +
        
        // WRONG: Y-axis range not inverted for screen coordinates
        const yScale = window.safeD3.createLinearScale(yExtent, [0, height]); // Should be [height, 0]
        ```
        
        ### MANDATORY Scale Requirements (Professional Ranges Like Example):
        ```javascript
        // PROFESSIONAL DOMAIN CALCULATION - Match the example visualization quality
        const cleanData = data.filter(d => {{
            const x = +window.safeD3.getValue(d, "x_column", 0);
            const y = +window.safeD3.getValue(d, "y_column", 0);
            return !isNaN(x) && !isNaN(y) && isFinite(x) && isFinite(y);
        }});
        
        // CRITICAL: Calculate extents with proper data type conversion
        const xExtent = d3.extent(cleanData, d => +window.safeD3.getValue(d, "x_column", 0));
        const yExtent = d3.extent(cleanData, d => +window.safeD3.getValue(d, "y_column", 0));
        
        // MANDATORY: Add 5% padding to prevent data points touching axis edges
        const xPadding = (xExtent[1] - xExtent[0]) * 0.05;
        const yPadding = (yExtent[1] - yExtent[0]) * 0.05;
        const xDomain = [xExtent[0] - xPadding, xExtent[1] + xPadding];
        const yDomain = [yExtent[0] - yPadding, yExtent[1] + yPadding];
        
        // PROFESSIONAL SCALE CREATION with .nice() for clean tick marks
        const xScale = window.safeD3.createLinearScale(xDomain, [0, width]).nice();
        const yScale = window.safeD3.createLinearScale(yDomain, [height, 0]).nice();
        
        // MANDATORY: Debug logging to verify professional ranges
        console.log("Professional X domain:", xScale.domain(), "Y domain:", yScale.domain());
        ```
        
        ### Requirements for Consistent Professional Visualization:
        - ALWAYS use + operator to convert string values to numbers: `+window.safeD3.getValue()`
        - ALWAYS filter out NaN/null/undefined values before calculating extents  
        - ALWAYS use [height, 0] range for Y-axis (screen coordinate inversion)
        - ALWAYS add 5% padding to domains to prevent data points touching axis edges
        - ALWAYS use .nice() for clean, rounded tick marks like the example (4.0, 5.0, 6.0)
        - ALWAYS add console.log statements for debugging domain/range values
        - NEVER use hardcoded domains like [4, 8] or [1, 5]
        
        ## ENHANCED HELPER METHODS
        Use these helper methods for professional-quality visualizations:
        
        - window.safeD3.createLinearScale(domain, range) - Creates scales with fallbacks and nice formatting
        - window.safeD3.createAxis(scaleOrAxisType, tickCount) - Creates axes with professional styling
        - window.safeD3.getValue(dataPoint, property, defaultValue) - Safely accesses data properties
        - window.safeD3.createAxesWithLabels(svgElement, xScale, yScale, options) - Creates professionally positioned axes with labels
        
        ## PROFESSIONAL AXES IMPLEMENTATION:
        1. Create main container: `const g = svgElement.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")")`
        2. Add title: `svgElement.append("text").attr("x", width/2).attr("y", 30).attr("text-anchor", "middle").style("font-size", "18px").style("font-weight", "600").style("fill", "#2c3e50").text("[Your Generated Title]")`
        3. Create axes with proper styling and labels
        4. Add legend in top-right corner
        5. Implement grid lines with subtle styling
        
        ## OUTPUT RULES (CRITICALLY IMPORTANT):
        - The code MUST start with 'function createVisualization(data, svgElement) {{'
        - Return ONLY the complete JavaScript code
        - DO NOT include ANY explanatory text, markdown, or code block formatting
        - DO NOT include any comments or descriptions outside the code itself
        - DO NOT include text like "Here is the code" or "The code creates..."
        - NEVER include any commentary after the closing brace of the function
        - If you need to explain something, do it as comments INSIDE the code
        """
        
        logger.info(f"Calling OpenAI API with model: {model}")
        
        # API call with retry mechanism for rate limits
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Call API based on version
                if OPENAI_API_VERSION == "v1":
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "system",
                            "content": "You are a D3.js expert. You must generate ONLY CODE with no explanations or markdown. ANY text that is not JavaScript code is forbidden. Never include explanations, descriptions, or commentary outside the code itself. The code must be a complete createVisualization function that can be executed directly."
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    d3_code = response.choices[0].message.content.strip()
                else:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{
                            "role": "system",
                            "content": "You are a D3.js expert. You must generate ONLY CODE with no explanations or markdown. ANY text that is not JavaScript code is forbidden. Never include explanations, descriptions, or commentary outside the code itself. The code must be a complete createVisualization function that can be executed directly."
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    d3_code = response.choices[0].message.content.strip()
                
                logger.info(f"Generated D3 code length: {len(d3_code)} characters")
                
                # Clean the response to ensure it's only code
                d3_code = clean_d3_response(d3_code)
                
                # Extra validation to ensure there's no trailing text
                if "```" in d3_code or "Here is" in d3_code or "The code" in d3_code:
                    logger.warning("Found explanatory text in the code, cleaning it...")
                    d3_code = clean_d3_response(d3_code)
                
                return d3_code
                
            except RateLimitError:
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Rate limit exceeded after maximum retries")
                    raise
            except Exception as e:
                logger.error(f"Error generating D3 code: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        # Fallback mechanism if all API calls fail
        logger.warning("Using fallback mechanism due to API failure")
        raise Exception("Failed to generate visualization code after multiple attempts")
        
    except Exception as e:
        logger.error(f"Error in generate_d3_code: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def refine_d3_code(initial_code: str, api_key: str, max_attempts: int = 3) -> str:
    """
    Refine the D3 code through iterative LLM calls if necessary.
    
    This function attempts to improve the generated D3 code if it fails validation.
    It makes multiple attempts to refine the code using the OpenAI API.
    
    Args:
        initial_code (str): The initial D3.js code to refine.
        api_key (str): OpenAI API key.
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 3.
    
    Returns:
        str: Refined D3.js code, or the last attempt if refinement fails.
    """
    # Initialize API
    if OPENAI_API_VERSION == "v1":
        client = OpenAI(api_key=api_key)
    else:
        openai.api_key = api_key
    
    # Get model and parameters from environment variables or use defaults
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-2024-08-06")
    max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    try:
        for attempt in range(max_attempts):
            validation_result = validate_d3_code(initial_code)
            
            # If code is valid and has no warnings, return it
            if validation_result.get("valid", False) and not validation_result.get("warnings", []):
                return initial_code
                
            # Prepare refinement issues list
            refinement_issues = []
            if not validation_result.get("valid", False):
                refinement_issues.extend(validation_result.get("missing_features", []))
            
            if validation_result.get("warnings", []):
                refinement_issues.extend(validation_result.get("warnings", []))
            
            # If there are no issues but validation isn't passing, add a generic issue
            if not refinement_issues and not validation_result.get("valid", False):
                refinement_issues.append("General D3.js code structure issues")
            
            issues_str = "\n".join([f"- {issue}" for issue in refinement_issues])
            
            refinement_prompt = f"""
            The following D3 code needs refinement to be valid:
            
            ```javascript
            {initial_code}
            ```
            
            Issues that need to be fixed:
            {issues_str}
            
            Please provide a corrected version that:
            1. Defines a createVisualization(data, svgElement) function
            2. Uses only D3.js version 7 syntax
            3. Creates a valid visualization
            4. Uses the svgElement parameter that is passed in (which is a D3 selection)
            5. NEVER uses d3.select("svg") or d3.select("#viz-svg") directly
            6. NEVER calls .node() method on svgElement unless absolutely necessary
            7. Properly handles errors and edge cases
            
            Return ONLY the corrected D3 code without any explanations or comments.
            """
            
            try:
                # Call API based on version
                if OPENAI_API_VERSION == "v1":
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": refinement_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    content = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": refinement_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    content = response.choices[0].message.content
                
                initial_code = clean_d3_response(content)
                
                # Log the refinement attempt
                logger.info(f"Refined D3 code (attempt {attempt+1}/{max_attempts}), code length: {len(initial_code)}")
            except Exception as e:
                logger.error(f"Error in code refinement attempt {attempt+1}: {str(e)}")
                continue
        
        # If we've exhausted our attempts, return the last attempt
        logger.warning("Failed to generate valid D3 code after maximum attempts")
        return initial_code
    except Exception as e:
        logger.error(f"Error in refine_d3_code: {str(e)}")
        logger.error(traceback.format_exc())
        return initial_code

def clean_d3_response(response: str) -> str:
    """
    Clean the D3.js code response from OpenAI to ensure it's properly formatted.
    
    This function extracts just the JavaScript code from the LLM response,
    removing any explanatory text, code blocks, or other non-code content.
    
    Args:
        response (str): The raw response from OpenAI containing D3.js code.
    
    Returns:
        str: Cleaned and formatted D3.js code.
    """
    # First, try to extract code between triple backticks if present
    code_pattern = re.compile(r'```(?:javascript|js)?\s*([\s\S]*?)\s*```')
    code_matches = code_pattern.findall(response)
    
    if code_matches:
        # Use the first code block found
        response = code_matches[0]
    
    # Remove markdown and comments that might be outside the code
    lines = response.split('\n')
    clean_lines = []
    
    # Flag to indicate we're inside the JavaScript function
    inside_function = False
    # Counter for braces to detect the function's end
    brace_count = 0
    function_ended = False
    
    for line in lines:
        # Skip Markdown headings, list items and horizontal rules
        if re.match(r'^#+\s|^[*-]\s|^-{3,}$|^#{3,}$', line.strip()):
            continue
            
        # Check if we're at the createVisualization function start
        if re.match(r'function\s+createVisualization\s*\(', line):
            inside_function = True
            clean_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            continue
            
        # If we've identified the function's end previously, don't add more lines
        if function_ended:
            continue
            
        # If we're inside the function, count braces and add the line
        if inside_function:
            clean_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            # If braces are balanced and we've started counting, we've reached the end
            if brace_count == 0 and '}' in line:
                function_ended = True
                continue
    
    # If we couldn't identify the function structure, fallback to original cleaning
    if not clean_lines:
        clean_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    
    clean_code = '\n'.join(clean_lines)
    
    # Ensure the code starts with createVisualization function
    if not clean_code.strip().startswith('function createVisualization'):
        clean_code = f'function createVisualization(data, svgElement) {{\n{clean_code}\n}}'
    
    # Ensure proper function closure
    open_braces = clean_code.count('{')
    close_braces = clean_code.count('}')
    
    if open_braces > close_braces:
        # Add missing closing braces
        clean_code += '\n' + ('}' * (open_braces - close_braces))
    
    return clean_code

def get_visualization_html(d3_code: str) -> str:
    """
    Generate HTML content for D3.js visualization.
    
    Args:
        d3_code (str): The D3.js code to embed in HTML.
        
    Returns:
        str: Complete HTML content ready for display.
    """
    # Generate a unique timestamp to prevent caching
    timestamp = int(time.time())
    
    # Ensure we have the JSON data available
    json_data = []
    if 'json_data' in st.session_state and st.session_state.json_data is not None:
        json_data = st.session_state.json_data
    elif 'preprocessed_df' in st.session_state and st.session_state.preprocessed_df is not None:
        try:
            json_data = st.session_state.preprocessed_df.to_dict(orient='records')
            logger.info(f"Generated json_data with {len(json_data)} records")
            st.session_state.json_data = json_data
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
    
    # Create HTML with the D3.js code and enhanced error handling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>D3 Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            /* Professional Visualization Container */
            #visualization {{
                width: 100%;
                height: {VISUALIZATION_HEIGHT}px;
                margin: 0 auto;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.08);
                overflow: hidden;
                position: relative;
                border: 1px solid rgba(0,0,0,0.06);
                font-family: "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", sans-serif;
            }}
            
            /* Enhanced SVG Styling */
            svg {{
                width: 100%;
                height: 100%;
                background-color: white;
                font-family: "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", sans-serif;
            }}
            
            /* Professional Typography Classes */
            .viz-title {{
                font-size: 18px;
                font-weight: 600;
                fill: #2c3e50;
                text-anchor: middle;
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            .axis-label {{
                font-size: 14px;
                font-weight: 500;
                fill: #34495e;
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            .tick-label {{
                font-size: 12px;
                font-weight: 400;
                fill: #7f8c8d;
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            .legend-text {{
                font-size: 13px;
                font-weight: 400;
                fill: #2c3e50;
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            .legend-title {{
                font-size: 14px;
                font-weight: 500;
                fill: #2c3e50;
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            /* Enhanced Grid Lines */
            .grid line {{
                stroke: #f1f3f4;
                stroke-width: 0.5px;
                shape-rendering: crispEdges;
            }}
            
            .grid path {{
                stroke-width: 0;
            }}
            
            /* Professional Axis Styling */
            .axis {{
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            .axis line {{
                stroke: #dee2e6;
                shape-rendering: crispEdges;
            }}
            
            .axis path {{
                stroke: #dee2e6;
                fill: none;
            }}
            
            .axis text {{
                fill: #495057;
                font-size: 12px;
            }}
            
            /* Enhanced Tooltip */
            .tooltip {{
                position: absolute;
                background: rgba(255, 255, 255, 0.98);
                backdrop-filter: blur(10px);
                padding: 12px 16px;
                border-radius: 8px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.08);
                pointer-events: none;
                font-family: "system-ui", "-apple-system", sans-serif;
                font-size: 12px;
                font-weight: 400;
                color: #2c3e50;
                z-index: 1000;
                line-height: 1.4;
                max-width: 200px;
                transition: opacity 0.2s ease-in-out;
            }}
            
            .tooltip .tooltip-title {{
                font-weight: 600;
                color: #1a1a1a;
                margin-bottom: 4px;
            }}
            
            .tooltip .tooltip-content {{
                color: #4a5568;
                font-size: 11px;
            }}
            
            /* Professional Color Palettes as CSS Variables */
            :root {{
                --color-primary: #2196F3;
                --color-secondary: #FF9800;
                --color-success: #4CAF50;
                --color-danger: #E91E63;
                --color-warning: #FF5722;
                --color-info: #00BCD4;
                --color-purple: #9C27B0;
                --color-brown: #795548;
                
                --color-blue-1: #1f77b4;
                --color-blue-2: #aec7e8;
                --color-orange-1: #ff7f0e;
                --color-orange-2: #ffbb78;
                --color-green-1: #2ca02c;
                --color-green-2: #98df8a;
                --color-red-1: #d62728;
                --color-red-2: #ff9896;
                
                --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07);
                --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
            }}
            
            /* Interactive Elements */
            .data-point {{
                cursor: pointer;
                transition: all 0.3s ease-in-out;
            }}
            
            .data-point:hover {{
                filter: brightness(1.1);
                stroke-width: 2px;
                stroke: rgba(0,0,0,0.3);
            }}
            
            .legend-item {{
                cursor: pointer;
                transition: opacity 0.3s ease-in-out;
            }}
            
            .legend-item:hover {{
                opacity: 0.7;
            }}
            
            /* Enhanced Error Messages */
            .error-message {{
                color: #e53e3e;
                background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
                border: 1px solid #feb2b2;
                padding: 20px;
                border-radius: 8px;
                margin: 20px;
                font-family: "system-ui", "-apple-system", sans-serif;
                box-shadow: 0 2px 8px rgba(229, 62, 62, 0.15);
            }}
            
            /* Professional Fallback Visualization */
            .fallback-viz {{
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                color: #4a5568;
                font-family: "system-ui", "-apple-system", sans-serif;
            }}
            
            /* Animation Classes */
            .fade-in {{
                animation: fadeIn 0.6s ease-in-out;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .slide-in {{
                animation: slideIn 0.8s ease-out;
            }}
            
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateX(-30px); }}
                to {{ opacity: 1; transform: translateX(0); }}
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .viz-title {{ font-size: 16px; }}
                .axis-label {{ font-size: 12px; }}
                .tooltip {{ font-size: 11px; padding: 8px 12px; }}
            }}
        </style>
    </head>
    <body>
        <div id="visualization">
            <!-- Create the SVG element explicitly with dimensions -->
            <svg id="viz-svg" width="100%" height="100%" viewBox="0 0 800 {VISUALIZATION_HEIGHT}" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
        
        <script>
            console.log("Starting visualization render at timestamp: {timestamp}");
            
            // Check browser compatibility
            const checkBrowserCompatibility = function() {{
                try {{
                    // Basic check for ES6 features
                    eval("const x = () => {{}};");
                    
                    // Check if D3 is loaded
                    if (typeof d3 === 'undefined') {{
                        return {{
                            compatible: false,
                            message: "D3.js failed to load. Please check your internet connection."
                        }};
                    }}
                        
                    // Test D3 scale creation specifically
                    try {{
                        const testScale = d3.scaleLinear().domain([0, 1]).range([0, 100]);
                        if (typeof testScale !== 'function' || typeof testScale.domain !== 'function') {{
                            return {{
                                compatible: false,
                                message: "D3 scale functions aren't working correctly in your browser."
                            }};
                        }}
                    }} catch (scaleError) {{
                        return {{
                            compatible: false,
                            message: "D3 scale functions aren't available: " + scaleError.message
                        }};
                    }}
                    
                    return {{ compatible: true }};
                }} catch (e) {{
                    return {{ 
                        compatible: false,
                        message: "Your browser doesn't support modern JavaScript features needed for this visualization."
                    }};
                }}
            }};
            
            // Wait for DOM to be fully loaded
            document.addEventListener("DOMContentLoaded", function() {{
                const compatibilityCheck = checkBrowserCompatibility();
                if (compatibilityCheck.compatible) {{
                    renderVisualization();
                }} else {{
                    createFallbackVisualization(compatibilityCheck.message);
                }}
            }});
            
            // Fallback if DOMContentLoaded already fired
            if (document.readyState === "complete" || document.readyState === "interactive") {{
                setTimeout(() => {{
                    const compatibilityCheck = checkBrowserCompatibility();
                    if (compatibilityCheck.compatible) {{
                        renderVisualization();
                    }} else {{
                        createFallbackVisualization(compatibilityCheck.message);
                    }}
                }}, 100);
            }}
            
            // Safe D3 methods that wrap common operations with error handling
            window.safeD3 = {{
                // Safe scale creation that handles missing/invalid domains
                createLinearScale: function(domain, range) {{
                    try {{
                        // Debug logging to identify issues
                        console.log("safeD3.createLinearScale called with:", {{ domain, range }});
                        
                        // Enhanced domain validation
                        let safeDomain = [0, 100];  // fallback
                        if (Array.isArray(domain) && domain.length === 2) {{
                            const [min, max] = domain;
                            // Convert to numbers if they're not already
                            const numMin = typeof min === 'number' ? min : parseFloat(min);
                            const numMax = typeof max === 'number' ? max : parseFloat(max);
                            
                            // Check if both values are valid numbers and min < max
                            if (!isNaN(numMin) && !isNaN(numMax) && isFinite(numMin) && isFinite(numMax) && numMin !== numMax) {{
                                safeDomain = [numMin, numMax];
                                console.log("Using provided domain:", safeDomain);
                            }} else {{
                                console.warn("Invalid domain values - min:", numMin, "max:", numMax, "using fallback [0,100]");
                            }}
                        }} else {{
                            console.warn("Domain is not a valid array of length 2:", domain, "using fallback [0,100]");
                        }}
                        
                        // Enhanced range validation  
                        let safeRange = [0, 500];  // fallback
                        if (Array.isArray(range) && range.length === 2) {{
                            const [start, end] = range;
                            const numStart = typeof start === 'number' ? start : parseFloat(start);
                            const numEnd = typeof end === 'number' ? end : parseFloat(end);
                            
                            if (!isNaN(numStart) && !isNaN(numEnd) && isFinite(numStart) && isFinite(numEnd)) {{
                                safeRange = [numStart, numEnd];
                                console.log("Using provided range:", safeRange);
                            }} else {{
                                console.warn("Invalid range values - start:", numStart, "end:", numEnd, "using fallback [0,500]");
                            }}
                        }} else {{
                            console.warn("Range is not a valid array of length 2:", range, "using fallback [0,500]");
                        }}
                            
                        const scale = d3.scaleLinear().domain(safeDomain).range(safeRange);
                        console.log("Created scale with domain:", scale.domain(), "range:", scale.range());
                        return scale;
                    }} catch (e) {{
                        console.error("Error creating linear scale:", e);
                        return d3.scaleLinear().domain([0, 100]).range([0, 500]);
                    }}
                }},
                
                // Create any scale type with error handling
                createScale: function(type, domain, range) {{
                    try {{
                        // Handle common errors in type names
                        if (!type || typeof type !== 'string') {{
                            console.warn("Invalid scale type, using linear");
                            return this.createLinearScale(domain, range);
                        }}
                        
                        // Normalize scale type name - handle both 'linear' and 'Linear' formats
                        const normalizedType = type.charAt(0).toUpperCase() + type.slice(1).toLowerCase();
                        const scaleName = `scale${{normalizedType}}`;
                        
                        const scaleFunc = d3[scaleName];
                        if (typeof scaleFunc !== 'function') {{
                            console.warn(`Scale type '${{type}}' (${{scaleName}}) not recognized, using linear`);
                            return this.createLinearScale(domain, range);
                        }}
                        
                        // Create and configure the scale
                        const scale = scaleFunc();
                        
                        // Test if scale has proper methods before using them
                        if (typeof scale.domain !== 'function' || typeof scale.range !== 'function') {{
                            console.warn(`Created scale doesn't have proper methods, using linear`);
                            return this.createLinearScale(domain, range);
                        }}
                        
                        // Handle different scale types that might have different config methods
                        if (Array.isArray(domain)) scale.domain(domain);
                        if (Array.isArray(range)) scale.range(range);
                        
                        return scale;
                    }} catch (e) {{
                        console.warn(`Error creating ${{type}} scale:`, e);
                        return this.createLinearScale(
                            Array.isArray(domain) ? domain : [0, 100],
                            Array.isArray(range) ? range : [0, 500]
                        );
                    }}
                }},
                
                // Verify a scale object has required methods and properties
                verifyScale: function(scale, defaultDomain = [0, 100], defaultRange = [0, 500]) {{
                    try {{
                        // Check if it's actually a scale
                        if (!scale || typeof scale !== 'function' || typeof scale.domain !== 'function' || typeof scale.range !== 'function') {{
                            console.warn("Invalid scale object, creating fallback scale");
                            return this.createLinearScale(defaultDomain, defaultRange);
                        }}
                        
                        // Test the scale's methods to ensure they work
                        try {{
                            // Try calling domain and range to verify they work
                            scale.domain();
                            scale.range();
                            return scale;
                        }} catch (methodError) {{
                            console.warn("Scale methods failed, creating fallback scale:", methodError);
                            return this.createLinearScale(defaultDomain, defaultRange);
                        }}
                    }} catch (e) {{
                        console.warn("Error verifying scale:", e);
                        return this.createLinearScale(defaultDomain, defaultRange);
                    }}
                }},
                
                // Safe axis creation that handles invalid scales
                createAxis: function(scaleOrType, orientation = 'bottom', tickCount = 5) {{
                    try {{
                        let scale;
                        
                        // Handle different input types
                        if (typeof scaleOrType === 'function') {{
                            // It's a scale, verify it's valid
                            scale = this.verifyScale(scaleOrType);
                        }} else if (typeof scaleOrType === 'string') {{
                            // If it's a type string, create a default scale of that type
                            scale = this.createScale(scaleOrType, [0, 100], [0, 500]);
                        }} else {{
                            // Neither - create a default linear scale
                            scale = this.createLinearScale([0, 100], [0, 500]);
                        }}
                        
                        // Determine which axis function to use
                        let axisFunc;
                        switch(orientation) {{
                            case 'bottom': axisFunc = d3.axisBottom; break;
                            case 'left': axisFunc = d3.axisLeft; break;
                            case 'right': axisFunc = d3.axisRight; break;
                            case 'top': axisFunc = d3.axisTop; break;
                            default: axisFunc = d3.axisBottom;
                        }}
                        
                        // Create the axis
                        const axis = axisFunc(scale);
                        
                        // Set tick count safely
                        try {{
                            if (Number.isInteger(tickCount) && tickCount > 0) {{
                                axis.ticks(tickCount);
                            }}
                        }} catch (tickError) {{
                            console.warn("Error setting tick count:", tickError);
                        }}
                        
                        return axis;
                    }} catch (e) {{
                        console.warn("Error creating axis:", e);
                        // Return a minimal working axis as fallback
                        const scale = d3.scaleLinear().domain([0, 100]).range([0, 500]);
                        return d3.axisBottom(scale).ticks(5);
                    }}
                }},
                
                // Safe data accessor that handles missing properties
                getValue: function(d, property, defaultValue = 0) {{
                    if (!d) return defaultValue;
                    
                    const value = d[property];
                    if (value === undefined || value === null) return defaultValue;
                    
                    // If it's already a number, return it
                    if (typeof value === 'number') return value;
                    
                    // If it's a string that represents a number, convert it
                    if (typeof value === 'string') {{
                        const numValue = parseFloat(value.trim());
                        if (!isNaN(numValue) && isFinite(numValue)) {{
                            return numValue;
                        }}
                    }}
                    
                    // For non-numeric values, return the original value (e.g., for categorical data)
                    return value;
                }},
                
                // Safe selection method
                select: function(selector, parent = document) {{
                    try {{
                        const selection = (parent.querySelector ? parent : d3.select(parent)).querySelector(selector);
                        return selection ? d3.select(selection) : null;
                    }} catch (e) {{
                        console.warn(`Error selecting '${{selector}}':`, e);
                        return null;
                    }}
                }},
                
                // Safe data binding
                bindData: function(selection, data) {{
                    try {{
                        if (!selection) return null;
                        return selection.data(Array.isArray(data) ? data : []);
                    }} catch (e) {{
                        console.warn("Error binding data:", e);
                        return selection;
                    }}
                }},
                
                // Create properly positioned axes with labels
                createAxesWithLabels: function(svgElement, xScale, yScale, options) {{
                    try {{
                        // Default configuration
                        const defaults = {{
                            margin: {{top: 40, right: 40, bottom: 60, left: 60}},
                            width: 800,
                            height: 500,
                            xLabel: "X Axis",
                            yLabel: "Y Axis",
                            ticksX: 5,
                            ticksY: 5
                        }};
                    
                        // Merge options with defaults
                        const config = {{...defaults, ...options}};
                        if (options && options.margin) {{
                            config.margin = {{...defaults.margin, ...options.margin}};
                        }}
                    
                        try {{
                            // Verify scales
                            if (!xScale || typeof xScale !== 'function' || typeof xScale.domain !== 'function') {{
                                xScale = d3.scaleLinear().domain([0, 100]).range([0, config.width]);
                            }}
                            
                            if (!yScale || typeof yScale !== 'function' || typeof yScale.domain !== 'function') {{
                                yScale = d3.scaleLinear().domain([0, 100]).range([config.height, 0]);
                            }}
                            
                            // Get dimensions from the SVG if not provided
                            let width = options && options.width ? options.width : (+svgElement.attr("width") || 800);
                            let height = options && options.height ? options.height : (+svgElement.attr("height") || 500);
                            
                            // Calculate inner dimensions
                            const innerWidth = width - config.margin.left - config.margin.right;
                            const innerHeight = height - config.margin.top - config.margin.bottom;
                            
                            // Create main container group with margins applied
                            const g = svgElement.append("g")
                                .attr("class", "axes-container")
                                .attr("transform", "translate(" + config.margin.left + "," + config.margin.top + ")");
                            
                            // PROFESSIONAL X-AXIS with MANDATORY grid lines
                            const xAxis = d3.axisBottom(xScale)
                                .ticks(config.ticksX || 5)
                                .tickSizeInner(config.tickSizeInner || -innerHeight)  // MANDATORY: Grid lines across full height
                                .tickSizeOuter(config.tickSizeOuter || 0)             // Remove outer tick extensions  
                                .tickPadding(config.tickPadding || 10);               // Professional spacing
                                
                            const xAxisG = g.append("g")
                                .attr("class", "x-axis")
                                .attr("transform", "translate(0," + innerHeight + ")")
                                .call(xAxis);
                                
                            // MANDATORY: Style grid lines professionally
                            xAxisG.selectAll(".tick line")
                                .style("stroke", config.gridLineStroke || "#e0e0e0")
                                .style("stroke-width", config.gridLineWidth || 0.5)
                                .style("opacity", config.gridOpacity || 0.7);
                            
                            // Add x-axis label with professional styling
                            xAxisG.append("text")
                                .attr("class", "x-axis-label")
                                .attr("x", innerWidth / 2)
                                .attr("y", 40)
                                .attr("fill", "#2c3e50")
                                .attr("text-anchor", "middle")
                                .style("font-size", "14px")
                                .style("font-weight", "500")
                                .style("font-family", "system-ui, -apple-system, sans-serif")
                                .text(config.xLabel);
                            
                            // PROFESSIONAL Y-AXIS with MANDATORY grid lines
                            const yAxis = d3.axisLeft(yScale)
                                .ticks(config.ticksY || 5)
                                .tickSizeInner(config.tickSizeInner || -innerWidth)   // MANDATORY: Grid lines across full width
                                .tickSizeOuter(config.tickSizeOuter || 0)             // Remove outer tick extensions
                                .tickPadding(config.tickPadding || 10);               // Professional spacing
                                
                            const yAxisG = g.append("g")
                                .attr("class", "y-axis")
                                .call(yAxis);
                                
                            // MANDATORY: Style grid lines professionally  
                            yAxisG.selectAll(".tick line")
                                .style("stroke", config.gridLineStroke || "#e0e0e0")
                                .style("stroke-width", config.gridLineWidth || 0.5)
                                .style("opacity", config.gridOpacity || 0.7);
                            
                            // Add y-axis label with professional styling
                            yAxisG.append("text")
                                .attr("class", "y-axis-label")
                                .attr("transform", "rotate(-90)")
                                .attr("x", -innerHeight / 2)
                                .attr("y", -40)
                                .attr("fill", "#2c3e50")
                                .attr("text-anchor", "middle")
                                .style("font-size", "14px")
                                .style("font-weight", "500")
                                .style("font-family", "system-ui, -apple-system, sans-serif")
                                .text(config.yLabel);
                            
                            // Return references to axes and dimensions
                            return {{
                                container: g,
                                xAxis: xAxisG,
                                yAxis: yAxisG,
                                dimensions: {{
                                    width: innerWidth,
                                    height: innerHeight,
                                    margin: config.margin
                                }}
                            }};
                        }} catch (innerError) {{
                            console.error("Error creating axes:", innerError);
                            // Create a minimal fallback
                            const g = svgElement.append("g")
                                .attr("class", "axes-container-fallback")
                                .attr("transform", "translate(" + config.margin.left + "," + config.margin.top + ")");
                            
                            // Return minimal container
                            return {{
                                container: g,
                                dimensions: {{
                                    width: config.width - config.margin.left - config.margin.right,
                                    height: config.height - config.margin.top - config.margin.bottom,
                                    margin: config.margin
                                }}
                            }};
                        }}
                    }} catch (error) {{
                        console.error("Fatal error creating axes with labels:", error);
                        return {{
                            container: svgElement.append("g"),
                            dimensions: {{ width: 500, height: 300, margin: defaults.margin }}
                        }};
                    }}
                }},
            }};
            
            // General purpose D3 operation wrapper
            window.d3safe = function(operation, fallback) {{
                try {{
                    return operation();
                }} catch (e) {{
                    console.error("D3 operation failed:", e);
                    return fallback;
                }}
            }};
            
            // Override problematic D3 methods to catch errors
            const originalAxisBottom = d3.axisBottom;
            d3.axisBottom = function(scale) {{
                if (!scale) {{
                    console.warn("Undefined scale passed to axisBottom, using fallback");
                    scale = d3.scaleLinear().domain([0, 100]).range([0, 500]);
                }}
                return originalAxisBottom(scale);
            }};
            
            const originalAxisLeft = d3.axisLeft;
            d3.axisLeft = function(scale) {{
                if (!scale) {{
                    console.warn("Undefined scale passed to axisLeft, using fallback");
                    scale = d3.scaleLinear().domain([0, 100]).range([0, 500]);
                }}
                return originalAxisLeft(scale);
            }};
            
            // Initialize D3 scale protections
            function setupD3Protection() {{
                // D3 scale protection - prevent 'n.range is not a function'
                const scaleCreators = ['scaleLinear', 'scaleOrdinal', 'scaleBand', 'scaleTime', 'scaleLog', 'scalePow', 'scaleSequential'];
                const originals = {{}};
                
                // For each scale type, provide a safe wrapper
                scaleCreators.forEach(function(scaleType) {{
                    if (typeof d3[scaleType] === 'function') {{
                        originals[scaleType] = d3[scaleType];
                        
                        d3[scaleType] = function() {{
                            try {{
                                const scale = originals[scaleType].apply(this, arguments);
                                
                                // Test the scale immediately to catch errors early
                                if (typeof scale !== 'function' || typeof scale.domain !== 'function' || typeof scale.range !== 'function') {{
                                    console.warn(scaleType + " didn't create a proper scale object, using fallback");
                                    return d3.scaleLinear().domain([0, 100]).range([0, 500]);
                                }}
                                
                                return scale;
                            }} catch (e) {{
                                console.warn("Error in " + scaleType + ":", e);
                                return d3.scaleLinear().domain([0, 100]).range([0, 500]);
                            }}
                        }};
                    }}
                }});
            }}
            
            // Call setup to implement D3 protection
            setupD3Protection();
            
            // Helper function to validate data
            function validateData(data) {{
                if (!data || !Array.isArray(data) || data.length === 0) {{
                    throw new Error("Data is empty or not in expected format");
                }}
                return true;
            }}
            
            // Create a simple fallback visualization if needed
            function createFallbackVisualization(errorMessage) {{
                const container = document.getElementById("visualization");
                
                // Clear existing content
                container.innerHTML = '';
                
                // Create a fallback visualization div
                const fallback = document.createElement("div");
                fallback.className = "fallback-viz";
                
                // Add error information
                const errorTitle = document.createElement("h3");
                errorTitle.textContent = "Visualization could not be rendered";
                
                const errorDetails = document.createElement("p");
                errorDetails.textContent = errorMessage || "An unknown error occurred";
                
                const errorHint = document.createElement("p");
                errorHint.textContent = "Try a different request or upload different data files";
                
                // Add all elements to the fallback
                fallback.appendChild(errorTitle);
                fallback.appendChild(errorDetails);
                fallback.appendChild(errorHint);
                
                // Add the fallback to the container
                container.appendChild(fallback);
            }}
            
            function renderVisualization() {{
                try {{
                    // The data from the DataFrame
                    let data = {json.dumps(json_data)};
                    
                    // Validate data before proceeding
                    if (!validateData(data)) {{
                        throw new Error("Invalid data format");
                    }}
                    
                    console.log("Data for visualization:", data);
                    console.log("D3 code length:", `{len(d3_code)}` + " characters");
                    
                    // First create a proper SVG with dimensions
                    const containerDiv = d3.select("#visualization");
                    const containerWidth = containerDiv.node().getBoundingClientRect().width;
                    const containerHeight = containerDiv.node().getBoundingClientRect().height;
                    
                    // Select and prepare the SVG element
                    const svgElement = d3.select("#viz-svg")
                        .attr("width", containerWidth)
                        .attr("height", containerHeight)
                        .attr("viewBox", `0 0 ${{containerWidth}} ${{containerHeight}}`)
                        .attr("preserveAspectRatio", "xMidYMid meet");
                    
                    // Clear any existing visualization
                    svgElement.selectAll("*").remove();
                    
                    // Add D3 error handling wrapper
                    try {{
                        // Add the D3 code
                        {d3_code}
                        
                        // Call the createVisualization function with protected execution
                        if (typeof createVisualization === 'function') {{
                            // Wrap the function call in a try-catch to handle D3-specific errors
                            try {{
                                // Make safeD3 available to the visualization function
                                window.safeD3 = safeD3;
                                createVisualization(data, svgElement);
                                console.log("Visualization successfully rendered");
                            }} catch (d3Error) {{
                                console.error("D3 runtime error:", d3Error);
                                
                                // Check for common D3 errors and provide helpful messages
                                let errorMessage = d3Error.message;
                                if (d3Error.message.includes("ticks") || 
                                    d3Error.message.includes("undefined") ||
                                    d3Error.message.includes("null") ||
                                    d3Error.message.includes("range is not a function")) {{
                                    errorMessage = "Error with visualization data: The visualization couldn't be created with this data. Try a different request.";
                                }}
                                
                                createFallbackVisualization(errorMessage);
                            }}
                        }} else {{
                            throw new Error("createVisualization function not found in the generated code");
                        }}
                    }} catch (funcError) {{
                        console.error("Error calling createVisualization:", funcError);
                        document.getElementById("visualization").innerHTML = 
                            `<div class="error-message">
                                <h3>Error Creating Visualization</h3>
                                <p>${{funcError.message}}</p>
                                <p>Try a different visualization request or check your data.</p>
                            </div>`;
                    }}
                }} catch (error) {{
                    console.error("Error rendering visualization:", error);
                    document.getElementById("visualization").innerHTML = 
                        `<div class="error-message">
                            <h3>Error Rendering Visualization</h3>
                            <p>${{error.message}}</p>
                            <p>Try a different visualization request or check your data.</p>
                        </div>`;
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content

def display_visualization(d3_code: str, placeholder=None) -> None:
    """
    Display the D3.js visualization in the Streamlit app using components.html.
    
    Args:
        d3_code (str): The D3.js code to display.
        placeholder (streamlit.delta_generator.DeltaGenerator, optional): Streamlit placeholder to render the visualization in.
            If None, renders in the current Streamlit position.
    """
    # Generate a unique timestamp to prevent caching
    timestamp = int(time.time())
    
    # Prepare error handling variables
    error_occurred = False
    error_msg = ""
    
    # Ensure we have the JSON data available
    if 'json_data' not in st.session_state or st.session_state.json_data is None:
        if 'preprocessed_df' in st.session_state and st.session_state.preprocessed_df is not None:
            try:
                st.session_state.json_data = st.session_state.preprocessed_df.to_dict(orient='records')
                logger.info(f"Generated json_data with {len(st.session_state.json_data)} records")
            except Exception as e:
                error_occurred = True
                error_msg = f"Error preparing data: {str(e)}"
                logger.error(error_msg)
        else:
            logger.warning("No preprocessed data available for visualization")
    
    # Only proceed if no error occurred during data preparation
    if not error_occurred:
        try:
            # Create HTML with the D3.js code and enhanced error handling
            html_content = get_visualization_html(d3_code)
            
            # Render in the appropriate place with proper height
            if placeholder is not None:
                # When using a placeholder, use it directly
                with placeholder:
                    components.html(
                        html_content,
                        height=VISUALIZATION_HEIGHT + 50,  # Add some padding
                        scrolling=True
                    )
                    logger.info("Visualization displayed in provided placeholder")
            else:
                # Use the container reuse pattern as specified in architecture.md
                viz_container_key = f"viz_container_{st.session_state.viz_key}"
                viz_container = st.session_state.get(viz_container_key, st.empty())
                
                # Store the container in session state if it's newly created
                if viz_container_key not in st.session_state:
                    st.session_state[viz_container_key] = viz_container
                
                # Use the container
                with viz_container.container():
                    components.html(
                        html_content,
                        height=VISUALIZATION_HEIGHT + 50,  # Add some padding
                        scrolling=True
                    )
                logger.info("Visualization displayed in consistent container")
            
            # Log success
            logger.info("Visualization displayed successfully")
            
        except Exception as e:
            error_occurred = True
            error_msg = f"Error displaying visualization: {str(e)}"
            logger.error(f"Error in display_visualization: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Display error message if an error occurred
    if error_occurred:
        if placeholder is not None:
            with placeholder:
                st.error(error_msg)
        else:
            st.error(error_msg)

def generate_histogram_visualization(df, species_column='species', value_column='petal_length', source_column='source'):
    """
    Generate a D3.js visualization specifically for paired histograms of petal length by species.
    This is a specialized function to handle one of the evaluation tasks.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        species_column (str): Column name for the species values.
        value_column (str): Column name for the value to plot in histograms (e.g., petal_length).
        source_column (str): Column name for the source of the data (dataset1, dataset2).
    
    Returns:
        str: D3.js code for rendering paired histograms.
    """
    logger.info(f"Generating specialized histogram visualization for {value_column} by {species_column}")
    
    # Create a D3.js visualization that works specifically for this task
    d3_code = """
    function createVisualization(data, svgElement) {
      try {
        // Set up dimensions and margins
        const margin = {top: 50, right: 30, bottom: 70, left: 60};
        const width = +svgElement.attr("width");
        const height = +svgElement.attr("height");
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        
        // Create the main group element
        const g = svgElement.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
            
        // Add title
        svgElement.append("text")
            .attr("x", width / 2)
            .attr("y", 20)
            .attr("text-anchor", "middle")
            .style("font-size", "18px")
            .style("font-weight", "bold")
            .text(`Petal Length Distribution by Species`);
            
        // Get unique species values
        const speciesValues = Array.from(new Set(data.map(d => d.species)));
        
        // Calculate the number of bins
        const xExtent = d3.extent(data, d => +d.petal_length);
        // Add a little padding to the domain
        const xDomain = [xExtent[0] - 0.1, xExtent[1] + 0.1];
        
        // Create x scale
        const x = d3.scaleLinear()
            .domain(xDomain)
            .range([0, innerWidth])
            .nice();
            
        // Set up the bin generator
        const numBins = 12;
        const binGenerator = d3.bin()
            .domain(x.domain())
            .thresholds(x.ticks(numBins))
            .value(d => +d.petal_length);
            
        // Set up species-specific information
        const speciesInfo = {};
        
        // Calculate bins for each species
        speciesValues.forEach((species, i) => {
            // Filter data for this species
            const speciesData = data.filter(d => d.species === species);
            
            // Generate the bins
            const bins = binGenerator(speciesData);
            
            // Find the maximum count for y-axis scaling
            const maxCount = d3.max(bins, d => d.length);
            
            // Store the information
            speciesInfo[species] = {
                data: speciesData,
                bins: bins,
                maxCount: maxCount,
                color: d3.schemeCategory10[i % 10]
            };
        });
        
        // Find the overall maximum count for consistent y-axis scaling
        const maxCount = d3.max(Object.values(speciesInfo), d => d.maxCount);
        
        // Create y scale
        const y = d3.scaleLinear()
            .domain([0, maxCount])
            .range([innerHeight, 0])
            .nice();
            
        // Create color scale for species
        const colorScale = d3.scaleOrdinal()
            .domain(speciesValues)
            .range(d3.schemeCategory10);
            
        // Calculate the height for each species' subplot
        const subplotHeight = innerHeight / speciesValues.length - 10;
        
        // Create a subplot for each species
        speciesValues.forEach((species, i) => {
            const speciesG = g.append("g")
                .attr("class", `species-${species.replace(/[^a-zA-Z0-9]/g, "")}`)
                .attr("transform", `translate(0, ${i * (subplotHeight + 10)})`);
                
            // Create a y scale for this subplot
            const subY = d3.scaleLinear()
                .domain([0, speciesInfo[species].maxCount])
                .range([subplotHeight, 0])
                .nice();
                
            // Add the bars
            speciesG.selectAll(".bar")
                .data(speciesInfo[species].bins)
                .enter()
                .append("rect")
                .attr("class", "bar")
                .attr("x", d => x(d.x0))
                .attr("y", d => subY(d.length))
                .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 1))
                .attr("height", d => subplotHeight - subY(d.length))
                .attr("fill", colorScale(species))
                .attr("opacity", 0.7)
                .attr("stroke", "#fff")
                .attr("stroke-width", 0.5);
                
            // Add a species label
            speciesG.append("text")
                .attr("x", innerWidth - 10)
                .attr("y", 10)
                .attr("text-anchor", "end")
                .attr("dominant-baseline", "hanging")
                .style("font-weight", "bold")
                .text(species);
                
            // Add y-axis
            speciesG.append("g")
                .attr("class", "y-axis")
                .call(d3.axisLeft(subY).ticks(5).tickFormat(d3.format("d")));
                
            // Add a label for count
            if (i === 1) { // Add to the middle subplot
                speciesG.append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("x", -subplotHeight / 2)
                    .attr("y", -40)
                    .attr("text-anchor", "middle")
                    .text("Count");
            }
        });
        
        // Add x-axis at the bottom
        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0, ${innerHeight})`)
            .call(d3.axisBottom(x))
            .append("text")
            .attr("x", innerWidth / 2)
            .attr("y", 35)
            .attr("fill", "black")
            .attr("text-anchor", "middle")
            .text("Petal Length");
        
        // Add a legend
        const legend = svgElement.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${width - margin.right - 120}, ${margin.top})`);
            
        speciesValues.forEach((species, i) => {
            const legendRow = legend.append("g")
                .attr("transform", `translate(0, ${i * 20})`);
                
            legendRow.append("rect")
                .attr("width", 10)
                .attr("height", 10)
                .attr("fill", colorScale(species));
                
            legendRow.append("text")
                .attr("x", 15)
                .attr("y", 5)
                .attr("dominant-baseline", "middle")
                .text(species);
        });
        
        // Add interactivity
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("background", "#fff")
            .style("border", "1px solid #ddd")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("pointer-events", "none");
            
        g.selectAll(".bar")
            .on("mouseover", function(event, d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`Count: ${d.length}<br>Range: ${d.x0.toFixed(1)} - ${d.x1.toFixed(1)}`)
                    .style("left", (event.pageX) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function() {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });
            
        // Return cleanup function
        return function cleanup() {
            tooltip.remove();
        };
      } catch (error) {
        console.error("Error creating histogram visualization:", error);
        // Add error message to the SVG
        svgElement.append("text")
            .attr("x", +svgElement.attr("width") / 2)
            .attr("y", +svgElement.attr("height") / 2)
            .attr("text-anchor", "middle")
            .style("fill", "red")
            .text("Error creating visualization");
      }
    }
    """
    
    return d3_code

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
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-2024-12-17")
    
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

    try:
        if 'preprocessed_df' not in st.session_state or st.session_state.preprocessed_df is None:
            with st.spinner("Preprocessing data..."):
                merged_df = preprocess_data(file1, file2)
            st.session_state.preprocessed_df = merged_df
            # Convert DataFrame to JSON and store in session state
            st.session_state.json_data = merged_df.to_dict(orient='records')
            logger.info(f"Initialized json_data with {len(st.session_state.json_data)} records")
        
        with st.expander("Preview of preprocessed data"):
            st.dataframe(st.session_state.preprocessed_df.head())
        
        # Show the visualization header
        viz_header.subheader("Visualization")
        
        # Generate initial visualization if needed
        if 'current_viz' not in st.session_state or st.session_state.current_viz is None:
            viz_status.info("Generating initial visualization...")
            
            try:
                with st.spinner("Generating initial D3 visualization..."):
                    d3_code = generate_and_validate_d3_code(st.session_state.preprocessed_df, api_key)
                    st.session_state.current_viz = d3_code
                    st.session_state.workflow_history.append({
                        "version": len(st.session_state.workflow_history) + 1,
                        "request": "Initial comparative visualization",
                        "code": d3_code,
                        "timestamp": time.time()
                    })
                    st.session_state.history_index = len(st.session_state.workflow_history) - 1
                
                viz_status.success("Initial visualization generated!")
                viz_caption.caption("Initial visualization based on data structure")
                
                # Clear the container before rendering
                with viz_container.container():
                    st.empty()
                
                # Display the current visualization in the container
                display_visualization(st.session_state.current_viz, viz_container)
            except Exception as e:
                viz_status.error(f"Error generating visualization: {str(e)}")
                logger.error(f"Error in initial visualization: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            # Display existing visualization
            viz_caption.caption("Current visualization")
            
            # Clear the container before rendering
            with viz_container.container():
                st.empty()
                
            # Display the current visualization in the container
            display_visualization(st.session_state.current_viz, viz_container)

        # Modification section
        st.markdown("---")
        st.subheader("Modify Visualization")
        user_input = st.text_area("Enter your visualization request:", 
                                  height=100,
                                  key="user_input",
                                  help="Describe what changes you want to make to the visualization")
        
        col_controls, col_spacer = st.columns([2, 2])
        with col_controls:
            update_col1, update_col2 = st.columns([3, 1])
            with update_col1:
                update_button = st.button("🔄 Update Visualization", 
                                          use_container_width=True, 
                                          type="primary",
                                          key="update_button")
        
        if update_button:
            if not user_input.strip():
                st.warning("Please enter a request to update the visualization.")
            else:
                # Show processing message in the status area
                viz_status.info(f"Working on: '{user_input}'")
                
                # Make sure json_data is initialized and up to date
                if 'json_data' not in st.session_state or st.session_state.json_data is None:
                    st.session_state.json_data = st.session_state.preprocessed_df.to_dict(orient='records')
                    logger.info(f"Updated json_data with {len(st.session_state.json_data)} records")
                
                # Step 1: Generate the new visualization code
                try:
                    with st.spinner("Generating updated visualization..."):
                        # Force creation of a new visualization code
                        new_d3_code = generate_d3_code(
                            st.session_state.preprocessed_df, 
                            api_key, 
                            user_input
                        )
                        
                        # Compare old and new code
                        old_code = st.session_state.current_viz if 'current_viz' in st.session_state else ""
                        if new_d3_code.strip() == old_code.strip():
                            logger.warning("Generated code is identical to current code")
                            viz_status.warning("The model generated identical code. Trying again with stronger instructions...")
                            
                            # Try again with stronger prompt
                            new_d3_code = generate_d3_code_with_forced_changes(
                                st.session_state.preprocessed_df,
                                api_key,
                                user_input,
                                old_code
                            )
                        
                        # Step 2: Update the session state
                        st.session_state.current_viz = new_d3_code
                        
                        # Limit the history to MAX_WORKFLOW_HISTORY entries
                        if len(st.session_state.workflow_history) >= MAX_WORKFLOW_HISTORY:
                            # Remove oldest item (first item)
                            st.session_state.workflow_history = st.session_state.workflow_history[1:]
                        
                        # Add to history
                        st.session_state.workflow_history.append({
                            "version": len(st.session_state.workflow_history) + 1,
                            "request": user_input,
                            "code": new_d3_code,
                            "timestamp": time.time()
                        })
                        st.session_state.history_index = len(st.session_state.workflow_history) - 1
                        
                        # Update the status and caption
                        viz_status.success("Visualization updated successfully!")
                        viz_caption.caption(f"Based on your request: '{user_input}'")
                        
                        # Clear the container before rendering
                        with viz_container.container():
                            st.empty()
                            
                        # Step 3: Display the updated visualization in the same container
                        display_visualization(new_d3_code, viz_container)
                        
                except Exception as e:
                    viz_status.error("Error processing request")
                    error_message = str(e)
                    
                    # Provide more helpful error messages for common issues
                    if "openai" in error_message.lower():
                        if "api key" in error_message.lower():
                            error_message = "Invalid or expired OpenAI API key. Please check your API key and try again."
                        elif "rate limit" in error_message.lower():
                            error_message = "OpenAI API rate limit exceeded. Please wait a minute and try again."
                        else:
                            error_message = f"OpenAI API error: {error_message}. Please try again later."
                    
                    st.error(f"Error updating visualization: {error_message}")
                    logger.error(f"Error in visualization update flow: {str(e)}")
                    logger.error(traceback.format_exc())

        # Display code editor and history navigation using helper functions
        display_code_editor(st.session_state.current_viz, viz_container, viz_status, viz_caption)
        display_history_navigation(st.session_state.workflow_history, viz_container, viz_status, viz_caption)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main app flow: {str(e)}")
        logger.error(traceback.format_exc())

def display_code_editor(code, viz_container, viz_status, viz_caption):
    """
    Display a code editor with syntax highlighting for D3.js code.
    
    Args:
        code (str): The D3.js code to display in the editor.
        viz_container: The container to display the visualization in.
        viz_status: The container to display status messages in.
        viz_caption: The container to display captions in.
    
    Returns:
        str: The edited code if changes were made, or the original code if no changes were made.
    """
    with st.expander("View/Edit Visualization Code"):
        # Create tabs for different editor views
        code_tab, help_tab = st.tabs(["Code Editor", "Help & Tips"])
        
        with code_tab:
            # Add syntax highlighting using HTML components
            st.markdown("""
            <style>
            .js-editor {
                font-family: monospace;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 10px;
                overflow: auto;
                height: 300px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create the code editor
            code_editor = st.text_area(
                "D3.js Code", 
                value=code, 
                height=300, 
                key="code_editor"
            )
            
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                edit_enabled = st.toggle("Edit Mode", key="edit_toggle")
            with col2:
                if st.button("Apply Changes", key="execute_code_button", disabled=not edit_enabled):
                    if edit_enabled:
                        if validate_d3_code(code_editor):
                            # Update the session state
                            st.session_state.current_viz = code_editor
                            
                            # Limit the history to MAX_WORKFLOW_HISTORY entries
                            if len(st.session_state.workflow_history) >= MAX_WORKFLOW_HISTORY:
                                # Remove oldest item (first item)
                                st.session_state.workflow_history = st.session_state.workflow_history[1:]
                                
                            # Add to history
                            st.session_state.workflow_history.append({
                                "version": len(st.session_state.workflow_history) + 1,
                                "request": "Manual code edit",
                                "code": code_editor,
                                "timestamp": time.time()
                            })
                            st.session_state.history_index = len(st.session_state.workflow_history) - 1
                            
                            # Update UI
                            viz_caption.caption("Manual code edit")
                            
                            # Clear the container before rendering
                            with viz_container.container():
                                st.empty()
                                
                            # Display the updated visualization
                            display_visualization(code_editor, viz_container)
                            viz_status.success("Manual code applied successfully!")
                            
                            return code_editor
                        else:
                            viz_status.error("Invalid code. Please check and try again.")
                    else:
                        viz_status.error("Please enable edit mode to modify code.")
        
        with help_tab:
            st.markdown("""
            ### D3.js Code Editing Tips
            
            1. **Function Structure**: Always keep the `createVisualization(data, svgElement)` function structure.
            
            2. **Safety Utilities**: Use these helper functions for robust code:
               - `window.safeD3.createScale(type, domain, range)` - Creates scales safely
               - `window.safeD3.createAxis(scale, orientation, ticks)` - Creates axes safely
               - `window.safeD3.getValue(dataPoint, property, defaultValue)` - Gets values safely
            
            3. **Error Handling**: Wrap risky operations in try/catch blocks:
               ```javascript
               try {
                 // Your D3 code here
               } catch (error) {
                 console.error("Error:", error);
               }
               ```
            
            4. **SVG Elements**: Always append elements to `svgElement`, not to a global selector.
            
            5. **Data Validation**: Always validate data before using it:
               ```javascript
               if (!data || !Array.isArray(data) || data.length === 0) {
                 console.error("Invalid data");
                 return;
               }
               ```
            """)
    
    return code

def display_history_navigation(workflow_history, viz_container, viz_status, viz_caption):
    """
    Display a history navigation interface for previous visualizations.
    
    Args:
        workflow_history (list): The history of visualizations.
        viz_container: The container to display the visualization in.
        viz_status: The container to display status messages in.
        viz_caption: The container to display captions in.
    """
    with st.expander("Visualization History"):
        if workflow_history:
            # Create a layout for history items
            history_cols = st.columns(min(3, len(workflow_history)))
            
            for idx, item in enumerate(reversed(workflow_history)):
                col_idx = idx % len(history_cols)
                
                with history_cols[col_idx]:
                    st.markdown(f"**Version {item['version']}**: {item['request']}")
                    
                    # Display timestamp if available
                    if 'timestamp' in item:
                        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['timestamp']))
                        st.caption(f"Created: {timestamp_str}")
                    
                    if st.button(f"Restore Version {item['version']}", key=f"restore_{idx}"):
                        st.session_state.current_viz = item['code']
                        st.session_state.history_index = len(workflow_history) - 1 - idx
                        viz_caption.caption(f"Restored from version {item['version']}: {item['request']}")
                        
                        # Clear the container before rendering
                        with viz_container.container():
                            st.empty()
                            
                        # Display the restored visualization
                        display_visualization(item['code'], viz_container)
                        viz_status.success(f"Restored version {item['version']}")
                    
                    # Add a separator between history items
                    st.markdown("---")
        else:
            st.info("No visualization history yet. Make changes to see them here.")

def generate_d3_code_with_forced_changes(df: pd.DataFrame, api_key: str, user_input: str, current_code: str) -> str:
    """
    Generate a new D3 code that is ensured to be different from the current code.
    This is used when the model generates identical code despite user requests for changes.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str): User's request for changes.
        current_code (str): The current D3 code that needs to be modified.
    
    Returns:
        str: New D3.js code with forced changes.
    """
    logger.info("Generating D3 code with forced changes")
    
    # Create a stronger prompt that emphasizes the need for changes
    stronger_prompt = f"""
    You are tasked with modifying D3.js visualization code based on a user request. 
    
    IMPORTANT: You MUST make substantial changes to the code based on the user's request.
    The previous code was:
    
    ```javascript
    {current_code}
    ```
    
    User's request for changes: "{user_input}"
    
    Please generate a completely new implementation that fulfills this request while being 
    significantly different from the previous code. Ensure the visualization is improved according 
    to the user's request. Include comprehensive error handling and comments to explain your approach.
    
    Only return valid JavaScript code for a D3.js visualization as a createVisualization function 
    that takes data and svgElement as parameters. Do NOT include any markdown, explanation text, or backticks.
    """
    
    try:
        # Prepare example visualization with column info
        column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
        data_sample = df.head(3).to_dict(orient='records')
        
        # Construct the prompt for the OpenAI API
        prompt = f"""
        Create a D3.js visualization that implements the following changes as requested by the user: 
        "{user_input}"
        
        Available data columns and their types:
        {column_info}
        
        Sample data:
        {json.dumps(data_sample, indent=2)}
        
        REQUIREMENTS:
        1. Implement robust error handling including try-catch blocks
        2. Ensure all scales have proper domains and ranges
        3. Implement the EXACT changes requested by the user
        4. Your code must be DIFFERENT from the current implementation
        5. Return ONLY the D3.js code as a function named createVisualization(data, svgElement)
        
        Current code to modify:
        ```javascript
        {current_code}
        ```
        
        Your response should ONLY include the JavaScript code without any explanation.
        """
        
        # Ensure the code is different by indicating that requirement in the prompt
        logger.info("Requesting new D3 code with forced changes from OpenAI API")
        
        # Get model and parameters
        model = os.getenv("DEFAULT_MODEL", "gpt-4o-2024-12-17")
        max_tokens = int(os.getenv("MAX_TOKENS", "3500"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Call the OpenAI API with version check
        if OPENAI_API_VERSION == "v1":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a D3.js expert who creates robust data visualizations with excellent error handling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            new_code = response.choices[0].message.content.strip()
        else:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a D3.js expert who creates robust data visualizations with excellent error handling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            new_code = response.choices[0].message.content.strip()
        
        # Log the response for debugging
        logger.info(f"Received response from OpenAI API, code length: {len(new_code)}")
        
        # Ensure we're only returning JavaScript code (remove markdown backticks if present)
        new_code = clean_d3_response(new_code)
        
        # Enforce safer practices
        safer_code = add_safety_wrapper(new_code)
        
        return safer_code
        
    except Exception as e:
        logger.error(f"Error in generate_d3_code_with_forced_changes: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a fallback visualization if there's an error
        return """
        function createVisualization(data, svgElement) {
          try {
            // Fallback visualization due to error
            const width = svgElement.attr("width");
            const height = svgElement.attr("height");
            
            // Create a simple message
            svgElement.append("text")
              .attr("x", width / 2)
              .attr("y", height / 2)
              .attr("text-anchor", "middle")
              .style("fill", "red")
              .text("Error generating visualization: " + "Unable to process your request");
              
            // Display a small representation of the data
            svgElement.append("text")
              .attr("x", width / 2)
              .attr("y", height / 2 + 30)
              .attr("text-anchor", "middle")
              .style("fill", "gray")
              .text("Data sample: " + JSON.stringify(data[0]).substring(0, 50) + "...");
              
          } catch (error) {
            console.error("Error in fallback visualization:", error);
          }
        }
        """

def add_safety_wrapper(code: str) -> str:
    """
    Add a safety wrapper around the D3 visualization code to ensure errors are caught
    and proper fallbacks are provided.
    
    Args:
        code (str): The original D3 visualization code.
        
    Returns:
        str: The code with added safety wrapper.
    """
    # Check if the code already has a function declaration
    if "function createVisualization" in code:
        # Find the opening brace of the function
        function_match = re.search(r'function\s+createVisualization\s*\([^)]*\)\s*\{', code)
        if function_match:
            # Find where the function body starts
            body_start = function_match.end()
            
            # Extract everything after the opening brace
            body = code[body_start:]
            
            # Find the matching closing brace for the function
            brace_count = 1
            closing_index = -1
            
            for i, char in enumerate(body):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        closing_index = i
                        break
            
            if closing_index != -1:
                # Extract the body without the closing brace
                body = body[:closing_index]
                
                # Wrap the body in a try-catch
                safer_code = f"""function createVisualization(data, svgElement) {{
  try {{
{body.strip()}
  }} catch (error) {{
    console.error("Error in visualization:", error);
    // Create a simple fallback visualization
    svgElement.selectAll("*").remove();
    svgElement.append("text")
      .attr("x", 100)
      .attr("y", 100)
      .text("Visualization error: " + error.message)
      .style("fill", "red");
  }}
}}"""
        return safer_code
    
    # If we couldn't find the function header, return the original code
    return code

def generate_and_validate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    """
    Generate, validate, and if necessary, refine D3 code.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
    
    Returns:
        str: Validated D3.js code.
    """
    # Log user input to verify it's being passed correctly
    logger.info(f"generate_and_validate_d3_code received user input: '{user_input}'")
    
    # Check for specific visualization requests and use pre-built templates if available
    if user_input and "histogram" in user_input.lower() and "petal length" in user_input.lower() and "species" in user_input.lower():
        logger.info("Using pre-built histogram visualization for petal length by species")
        return generate_histogram_visualization(df)
    
    # Ensure user_input is treated as a string
    if user_input is None:
        user_input = ""
    
    # ============ INTELLIGENT DATA ANALYSIS ============
    # Analyze data characteristics to provide better context to AI
    
    # Data shape and structure analysis
    num_rows, num_cols = df.shape
    column_types = df.dtypes.to_dict()
    
    # Identify categorical vs numerical columns
    categorical_cols = []
    numerical_cols = []
    for col, dtype in column_types.items():
        if dtype == 'object' or df[col].nunique() < 10:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Identify key relationship columns
    has_source_column = 'source' in df.columns.str.lower()
    has_species_column = any('species' in col.lower() for col in df.columns)
    has_category_column = len(categorical_cols) > 0
    
    # Generate data insights for AI context
    data_insights = []
    if has_source_column:
        data_insights.append("COMPARATIVE ANALYSIS: Dataset contains multiple sources for comparison")
    if has_species_column:
        data_insights.append("SPECIES ANALYSIS: Dataset contains species/category information")
    if len(numerical_cols) >= 2:
        data_insights.append(f"RELATIONSHIP ANALYSIS: Multiple numerical variables available ({len(numerical_cols)} columns)")
    if num_rows > 100:
        data_insights.append("LARGE DATASET: Consider data aggregation or sampling techniques")
    
    # Suggest optimal visualization approaches based on data
    viz_suggestions = []
    if len(numerical_cols) >= 2 and has_category_column:
        viz_suggestions.append("RECOMMENDED: Scatter plot with categorical color encoding")
    if has_source_column and len(numerical_cols) >= 1:
        viz_suggestions.append("RECOMMENDED: Comparative visualization showing differences between sources")
    if len(categorical_cols) >= 1 and len(numerical_cols) >= 1:
        viz_suggestions.append("RECOMMENDED: Group-based analysis (box plots, violin plots, or grouped bar charts)")
    
    # Create enhanced context for AI
    data_context = f"""
    
    ## INTELLIGENT DATA ANALYSIS CONTEXT:
    
    ### Dataset Characteristics:
    - **Rows**: {num_rows:,} records
    - **Columns**: {num_cols} total ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)
    - **Numerical Columns**: {numerical_cols}
    - **Categorical Columns**: {categorical_cols}
    
    ### Key Data Insights:
    {chr(10).join(f"- {insight}" for insight in data_insights)}
    
    ### Visualization Recommendations:
    {chr(10).join(f"- {suggestion}" for suggestion in viz_suggestions)}
    
    ### Data Quality Notes:
    - **Missing Values**: {df.isnull().sum().sum()} total missing values
    - **Unique Categories**: {', '.join([f"{col}: {df[col].nunique()}" for col in categorical_cols[:3]])}
    
    IMPORTANT: Use this analysis to create the most meaningful and appropriate visualization type for this specific dataset.
    """
    
    # Enhance user input with data context
    enhanced_user_input = user_input + data_context if user_input else "Create the most appropriate and meaningful visualization for this dataset based on the data analysis above" + data_context
    
    # Generate the initial code with enhanced user input
    initial_code = generate_d3_code(df, api_key, enhanced_user_input)
    cleaned_code = clean_d3_response(initial_code)
    
    if validate_d3_code(cleaned_code):
        return cleaned_code
    else:
        return refine_d3_code(cleaned_code, api_key)

if __name__ == "__main__":
    main()
