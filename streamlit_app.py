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

# Define MAX_WORKFLOW_HISTORY constant
MAX_WORKFLOW_HISTORY = 20

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
    
    # Check for basic D3 v7 method calls
    d3_methods = ['d3.select', 'd3.scaleLinear', 'd3.axisBottom', 'd3.axisLeft']
    if not any(method in code for method in d3_methods):
        missing_features.append("Basic Structure: D3 method calls")
    
    # Check for balanced braces
    if code.count('{') != code.count('}'):
        missing_features.append("Basic Structure: Balanced braces")
    
    # Check for potential SVG element handling issues
    if 'd3.select("svg")' in code or 'd3.select("#viz-svg")' in code:
        warnings.append("DOM Manipulation: Direct SVG selection instead of using provided svgElement")
    
    if re.search(r'svgElement\.node\(\)', code):
        warnings.append("DOM Manipulation: Using svgElement.node() which may cause issues")
    
    if re.search(r'setAttribute\s*\(', code) and not re.search(r'\.attr\s*\(', code):
        warnings.append("DOM Manipulation: Using setAttribute directly instead of D3's attr method")
    
    # Check if code appends to body instead of svgElement
    if 'd3.select("body")' in code:
        warnings.append("DOM Structure: Appending to body instead of svgElement")
    
    # Check for error handling
    if not any(term in code for term in ['try {', 'catch (', 'if (!data', 'if (data', 'data.length', '=== 0', '== 0', '=== undefined', '== undefined']):
        warnings.append("Error Handling: No basic data validation or error checking")
    
    # Return dictionary with validation results
    return {
        "valid": len(missing_features) == 0,
        "missing_features": missing_features,
        "warnings": warnings
    }

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
        
        elif category == "Interactivity":
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
        model = os.getenv("DEFAULT_MODEL", "gpt-4")
        max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Enhanced prompt with quality standards matching the sample
        prompt = f"""
        # D3.js VISUALIZATION CREATION

        Create a high-quality D3.js version 7 visualization based on the following:
        
        ## USER REQUEST:
        {user_input if user_input else "Create an initial visualization that best represents this data"}
        
        ## DATA INFORMATION:
        Schema: {schema_str}
        
        Sample data: 
        ```json
        {json.dumps(data_sample[:5], indent=2)}
        ```
        
        ## REQUIREMENTS:
        1. Create a function named createVisualization(data, svgElement) that follows professional D3 standards
           - IMPORTANT: svgElement parameter is a D3 selection, not a raw DOM element
           - Always use svgElement.append() instead of d3.select("svg").append()
           - Do NOT use .node() on svgElement unless absolutely necessary
        
        2. Include a comprehensive configuration object with:
           - Proper margins (top, right, bottom, left)
           - Width and height derived from svgElement
           - Transition durations and easing functions
           - Color scales
           - Tooltip settings
           - Animation parameters
        
        3. Implement responsive design:
           - Get container dimensions from svgElement using .attr("width") and .attr("height")
           - Use viewBox for SVG scaling
           - Handle window resize events
           - Add preserveAspectRatio
        
        4. Create professional-looking axes:
           - Properly styled grid lines
           - Formatted tick values
           - Rotated labels if needed
           - Smooth transitions for updates
        
        5. Add rich interactivity:
           - Detailed tooltips with all relevant data
           - Smooth transitions and animations
           - Highlight effects on hover
           - Click interactions for additional details
           - Zoom and brush functionality if appropriate
        
        6. Include accessibility features:
           - ARIA attributes
           - Role descriptions
           - Keyboard navigation if applicable
        
        7. Add comprehensive error handling:
           - Check for data existence and structure
           - Provide fallbacks for missing values
           - Visual feedback for errors
        
        The code must start with 'function createVisualization(data, svgElement) {{'
        Return ONLY the complete JavaScript code with no explanations.
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
                            "content": "You are a D3.js expert. Generate only professional, production-ready visualization code with no explanations or markdown. Your code should be comprehensive, well-structured, include detailed configuration options, responsive design, smooth animations, rich interactivity, accessibility features, and thorough error handling."
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
                            "content": "You are a D3.js expert. Generate only professional, production-ready visualization code with no explanations or markdown. Your code should be comprehensive, well-structured, include detailed configuration options, responsive design, smooth animations, rich interactivity, accessibility features, and thorough error handling."
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    d3_code = response.choices[0].message.content.strip()
                
                logger.info(f"Generated D3 code length: {len(d3_code)} characters")
                
                # Validate the generated code has the required function
                if not d3_code.startswith("function createVisualization"):
                    logger.warning("Generated code doesn't start with createVisualization function, fixing...")
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
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")
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
    Clean the LLM response to ensure it only contains valid D3 code.
    
    This function removes markdown formatting, non-JavaScript lines,
    and ensures the code starts with the createVisualization function
    and has proper function closure.
    
    Args:
        response (str): The raw response from the LLM.
    
    Returns:
        str: Cleaned D3.js code with valid function structure.
    """
    # Remove any potential markdown code blocks
    response = response.replace("```javascript", "").replace("```js", "").replace("```", "")
    
    # Remove any lines that don't look like JavaScript
    clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
    clean_code = '\n'.join(clean_lines)
    
    # Check if code already has createVisualization function
    if not clean_code.strip().startswith('function createVisualization'):
        # If not, wrap the entire code in the function
        clean_code = f'function createVisualization(data, svgElement) {{\n{clean_code}\n}}'
    
    # Ensure proper function closure
    open_braces = clean_code.count('{')
    close_braces = clean_code.count('}')
    
    if open_braces > close_braces:
        # Add missing closing braces
        clean_code += '\n' + ('}' * (open_braces - close_braces))
    
    return clean_code

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
    
    try:
        # Ensure we have the JSON data available
        if 'json_data' not in st.session_state or st.session_state.json_data is None:
            if 'preprocessed_df' in st.session_state and st.session_state.preprocessed_df is not None:
                st.session_state.json_data = st.session_state.preprocessed_df.to_dict(orient='records')
                logger.info(f"Generated json_data with {len(st.session_state.json_data)} records")
            else:
                raise ValueError("No data available for visualization")
        
        # Create HTML with the D3.js code and debugging
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                #visualization {{
                    width: 100%;
                    height: 550px;
                    overflow: hidden;
                    margin: 0;
                    padding: 0;
                }}
                
                svg {{
                    width: 100%;
                    height: 100%;
                    background-color: white;
                }}
                
                .tooltip {{
                    position: absolute;
                    background: rgba(255, 255, 255, 0.95);
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.25);
                    pointer-events: none;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    z-index: 10;
                }}
                
                .error-message {{
                    color: #d9534f;
                    padding: 20px;
                    border: 1px solid #d9534f;
                    border-radius: 5px;
                    background-color: #f9f2f2;
                    margin: 20px;
                    font-family: Arial, sans-serif;
                }}
            </style>
        </head>
        <body>
            <div id="visualization">
                <!-- Create the SVG element explicitly with dimensions -->
                <svg id="viz-svg" width="100%" height="100%"></svg>
            </div>
            
            <script>
                console.log("Starting visualization render at timestamp: {timestamp}");
                
                // Wait for DOM to be fully loaded
                document.addEventListener("DOMContentLoaded", function() {{
                    renderVisualization();
                }});
                
                // Fallback if DOMContentLoaded already fired
                if (document.readyState === "complete" || document.readyState === "interactive") {{
                    setTimeout(renderVisualization, 100);
                }}
                
                function renderVisualization() {{
                    try {{
                        // The data from the DataFrame
                        const data = {json.dumps(st.session_state.json_data)};
                        
                        // Debug data and code
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
                        
                        // Add the D3 code
                        {d3_code}
                        
                        // Call the createVisualization function
                        try {{
                            if (typeof createVisualization === 'function') {{
                                // Ensure we're passing a proper D3 selection, not a raw DOM element
                                createVisualization(data, svgElement);
                                console.log("Visualization successfully rendered");
                            }} else {{
                                throw new Error("createVisualization function not found in the generated code");
                            }}
                        }} catch (funcError) {{
                            console.error("Error calling createVisualization:", funcError);
                            document.getElementById("visualization").innerHTML = 
                                `<div class="error-message">
                                    <h3>Error Executing Visualization Function</h3>
                                    <p>${{funcError.message}}</p>
                                    <pre>${{funcError.stack}}</pre>
                                </div>`;
                        }}
                    }} catch (error) {{
                        console.error("Error rendering visualization:", error);
                        document.getElementById("visualization").innerHTML = 
                            `<div class="error-message">
                                <h3>Error Rendering Visualization</h3>
                                <p>${{error.message}}</p>
                                <pre>${{error.stack}}</pre>
                            </div>`;
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        # Render in the appropriate place
        if placeholder is not None:
            # When using a placeholder, we need to use the correct method
            with placeholder.container():
                # Use components.html inside the placeholder's container
                components.html(
                    html_content,
                    height=600,
                    scrolling=True
                )
        else:
            # Use components.html to display the visualization with proper height
            components.html(
                html_content,
                height=600,
                scrolling=True
            )
        
        # Log success
        logger.info("Visualization displayed successfully")
        
    except Exception as e:
        error_msg = f"Error displaying visualization. Please check the browser console for details. Error: {str(e)}"
        logger.error(f"Error in display_visualization: {str(e)}")
        logger.error(traceback.format_exc())
        
        if placeholder is not None:
            placeholder.error(error_msg)
        else:
            st.error(error_msg)

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
    
    # Ensure user_input is treated as a string
    if user_input is None:
        user_input = ""
    
    # Generate the initial code with user input
    initial_code = generate_d3_code(df, api_key, user_input)
    cleaned_code = clean_d3_response(initial_code)
    
    if validate_d3_code(cleaned_code):
        return cleaned_code
    else:
        return refine_d3_code(cleaned_code, api_key)


def main():
    st.set_page_config(page_title="🎨 Comparative Visualization Generator", page_icon="✨", layout="wide")
    st.title("🎨 Comparative Visualization Generator")

    # Get API key from environment, secrets, or user input
    api_key = get_api_key()
    
    # Initialize session state for workflow history if it doesn't exist
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []
    
    # Display model information in a less prominent place if needed
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")
    
    st.header("Upload CSV Files")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload first CSV file", type="csv")
    with col2:
        file2 = st.file_uploader("Upload second CSV file", type="csv")

    if 'update_viz' not in st.session_state:
        st.session_state.update_viz = False

    # Create visualization containers - defining these outside the file check
    # ensures they remain stable even as the UI updates
    viz_header = st.empty()  # Header container
    viz_status = st.empty()  # For showing viz status
    viz_caption = st.empty()  # For showing viz description
    viz_container = st.empty()  # The main visualization container

    if file1 and file2:
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
                with st.spinner("Generating initial D3 visualization..."):
                    d3_code = generate_and_validate_d3_code(st.session_state.preprocessed_df, api_key)
                    st.session_state.current_viz = d3_code
                    st.session_state.workflow_history.append({
                        "version": len(st.session_state.workflow_history) + 1,
                        "request": "Initial comparative visualization",
                        "code": d3_code
                    })
                viz_status.empty()
                viz_caption.caption("Initial visualization based on data structure")
            
            # Clear the container before rendering to avoid stacking issues
            with viz_container.container():
                st.empty()
                
            # Display the current visualization in the container
            display_visualization(st.session_state.current_viz, viz_container)

            # Modification section
            st.markdown("---")
            st.subheader("Modify Visualization")
            user_input = st.text_area("Enter your visualization request:", 
                                      height=100,
                                      help="Describe what changes you want to make to the visualization")
            
            update_col1, update_col2 = st.columns([3, 1])
            with update_col1:
                update_button = st.button("🔄 Update Visualization", use_container_width=True, type="primary")
            
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
                            if new_d3_code == old_code:
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
                            
                            # Add to history
                            st.session_state.workflow_history.append({
                                "version": len(st.session_state.workflow_history) + 1,
                                "request": user_input,
                                "code": new_d3_code
                            })
                            
                            # Update the status and caption
                            viz_status.success("Visualization updated successfully!")
                            viz_caption.caption(f"Based on your request: '{user_input}'")
                            
                            # Clear the container before rendering to avoid stacking issues
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

            # Code editor section
            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area("D3.js Code", value=st.session_state.current_viz, height=300, key="code_editor")
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    edit_enabled = st.toggle("Edit", key="edit_toggle")
                with col2:
                    if st.button("Execute Code"):
                        if edit_enabled:
                            if validate_d3_code(code_editor):
                                st.session_state.current_viz = code_editor
                                st.session_state.workflow_history.append({
                                    "version": len(st.session_state.workflow_history) + 1,
                                    "request": "Manual code edit",
                                    "code": code_editor
                                })
                                viz_caption.caption("Manual code edit")
                                
                                # Clear the container before rendering
                                with viz_container.container():
                                    st.empty()
                                    
                                display_visualization(code_editor, viz_container)
                                viz_status.success("Manual code applied successfully!")
                            else:
                                viz_status.error("Invalid code. Please check and try again.")
                        else:
                            viz_status.error("Please enable edit mode to modify code.")
            
            # Visualization history
            with st.expander("Visualization History"):
                if st.session_state.workflow_history:
                    for idx, item in enumerate(reversed(st.session_state.workflow_history)):
                        st.markdown(f"**Version {item['version']}**: {item['request']}")
                        if st.button(f"Restore Version {item['version']}", key=f"restore_{idx}"):
                            st.session_state.current_viz = item['code']
                            viz_caption.caption(f"Restored from version {item['version']}: {item['request']}")
                            
                            # Clear the container before rendering
                            with viz_container.container():
                                st.empty()
                                
                            display_visualization(item['code'], viz_container)
                            viz_status.success(f"Restored visualization from version {item['version']}")
                else:
                    st.info("No visualization history available yet.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Application error: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        # Show instructions when no files are uploaded
        viz_header.empty()  # Clear the visualization header
        viz_status.empty()  # Clear any status messages
        viz_caption.empty()  # Clear any captions
        viz_container.empty()  # Clear the visualization container
        
        st.info("📊 Please upload both CSV files to generate a visualization")
        st.markdown("""
        ### How to use this app:
        1. Enter your OpenAI API key above
        2. Upload two CSV files with similar schema
        3. The app will generate an initial visualization
        4. Type your modification requests in the text box
        5. Click "Update Visualization" to apply changes
        """)
        
        # Clear any existing visualizations if files are removed
        if 'current_viz' in st.session_state:
            st.session_state.current_viz = None
        if 'preprocessed_df' in st.session_state:
            st.session_state.preprocessed_df = None
        if 'json_data' in st.session_state:
            st.session_state.json_data = None

def generate_d3_code_with_forced_changes(df, api_key, user_input, current_code):
    """
    Generate D3.js code with forced changes when the regular generation
    produces identical code to what's currently displayed.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str): User's request for visualization modifications.
        current_code (str): The current D3.js code being displayed.
    
    Returns:
        str: New D3.js code with forced changes.
    """
    if not user_input or user_input.strip() == "":
        logger.warning("Empty user input for forced changes, returning current code")
        return current_code
    
    data_sample = df.head(5).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
    
    # Initialize OpenAI client based on version
    if OPENAI_API_VERSION == "v1":
        client = OpenAI(api_key=api_key)
    else:
        openai.api_key = api_key
    
    # Get model and parameters
    model = os.getenv("DEFAULT_MODEL", "gpt-4")
    max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
    temperature = float(os.getenv("TEMPERATURE", "0.9"))  # Higher temperature for more variability
    
    # Create a prompt that explicitly requests significant changes
    prompt = f"""
    # D3.js VISUALIZATION REDESIGN - SIGNIFICANT CHANGES REQUIRED
    
    The current visualization needs significant changes based on this user request:
    
    ## USER REQUEST (MUST BE ADDRESSED):
    {user_input}
    
    ## CURRENT CODE (MUST BE CHANGED):
    ```javascript
    {current_code}
    ```
    
    ## DATA INFORMATION:
    Schema: {schema_str}
    
    Sample data: 
    ```json
    {json.dumps(data_sample[:5], indent=2)}
    ```
    
    ## REQUIREMENTS:
    1. Create a COMPLETELY DIFFERENT visualization that fulfills the user request
       - IMPORTANT: The svgElement parameter is a D3 selection, not a raw DOM element
       - Always use svgElement.append() instead of d3.select("svg").append()
       - Do NOT use .node() on svgElement unless absolutely necessary
       
    2. Do NOT return code similar to the current code
    
    3. Change the visualization type, layout, or core approach
    
    4. Implement responsive design:
       - Get container dimensions from svgElement using .attr("width") and .attr("height")
       - Use viewBox for SVG scaling
       - Adapt the visualization to the container size
    
    5. Add detailed comments explaining your visualization logic
    
    The code must start with 'function createVisualization(data, svgElement) {{'
    Return ONLY the complete JavaScript code.
    """
    
    logger.info("Using forced change prompt due to identical code generation")
    
    try:
        # Implement retry mechanism for API calls
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
                            "content": "You are a D3.js expert. The user needs a COMPLETELY NEW visualization that is significantly different from their current one. Be creative and make substantial changes."
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
                            "content": "You are a D3.js expert. The user needs a COMPLETELY NEW visualization that is significantly different from their current one. Be creative and make substantial changes."
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    d3_code = response.choices[0].message.content.strip()
                
                d3_code = clean_d3_response(d3_code)
                logger.info(f"Generated new D3 code with forced changes, length: {len(d3_code)} characters")
                
                # Verify the new code is actually different
                if d3_code.strip() == current_code.strip():
                    logger.warning("Generated code is still identical, retrying with higher temperature")
                    temperature += 0.1  # Increase temperature for more variability
                    continue
                
                return d3_code
                
            except RateLimitError:
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Rate limit exceeded after maximum retries")
                    return current_code
            except Exception as e:
                logger.error(f"Error in forced code generation: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        # If all attempts failed, return current code with a warning message
        logger.error("Failed to generate different code after multiple attempts")
        return current_code
    except Exception as e:
        logger.error(f"Error in forced code generation: {str(e)}")
        logger.error(traceback.format_exc())
        return current_code

if __name__ == "__main__":
    main()
