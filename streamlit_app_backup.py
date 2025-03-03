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
    Generate D3.js visualization code based on DataFrame and user input.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
    
    Returns:
        str: D3.js code as a string.
    """
    # Ensure user_input is treated as a string
    if user_input is None:
        user_input = ""
    
    logger.info(f"Generating D3 code with user input: '{user_input}'")
    
    try:
        # Prepare schema information for the prompt
        columns_info = {}
        for col in df.columns:
            # Get type information and basic stats
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            has_nulls = df[col].isna().any()
            
            # Detect numerical vs categorical
            if pd.api.types.is_numeric_dtype(dtype):
                min_val = df[col].min()
                max_val = df[col].max()
                columns_info[col] = {
                    "type": "numeric",
                    "dtype": str(dtype),
                    "min": float(min_val) if not pd.isna(min_val) else None,
                    "max": float(max_val) if not pd.isna(max_val) else None,
                    "unique_count": int(unique_count),
                    "has_nulls": bool(has_nulls)
                }
            else:
                sample_values = df[col].dropna().unique()[:5].tolist()
                columns_info[col] = {
                    "type": "categorical",
                    "dtype": str(dtype),
                    "sample_values": sample_values,
                    "unique_count": int(unique_count),
                    "has_nulls": bool(has_nulls)
                }
        
        # Get sample data
        sample_data = df.head(5).to_dict(orient='records')
        
        # Create enhanced visualization template with improved D3 practices
        visualization_template = """
// Enhanced visualization template for comparative data visualization
function createVisualization(data, svgElement) {
  try {
    // Extract dimensions and set up margins following D3 best practices
    const width = parseInt(svgElement.attr("width"));
    const height = parseInt(svgElement.attr("height"));
    const margin = {top: 50, right: 80, bottom: 60, left: 60};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create the inner visualization group with proper transform
    const g = svgElement.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Data validation - ensure we have data to work with
    if (!data || !Array.isArray(data) || data.length === 0) {
      throw new Error("No data available for visualization");
    }
    
    // Detect categorical vs numerical columns for better visualization decisions
    const categoricalColumns = [];
    const numericalColumns = [];
    
    // Process the first data item to categorize columns
    const firstItem = data[0];
    Object.keys(firstItem).forEach(key => {
      const value = firstItem[key];
      if (typeof value === 'number') {
        numericalColumns.push(key);
      } else {
        categoricalColumns.push(key);
      }
    });
    
    // Determine potential color-coding column (usually categorical with few unique values)
    let colorColumn = null;
    if (categoricalColumns.length > 0) {
      // Find the categorical column with the fewest unique values (good for color-coding)
      const uniqueCounts = {};
      categoricalColumns.forEach(col => {
        const uniqueValues = new Set(data.map(d => d[col]));
        uniqueCounts[col] = uniqueValues.size;
      });
      
      // Sort by unique count and get the first one (with at least 2 but not too many values)
      const sortedColumns = Object.entries(uniqueCounts)
        .filter(([_, count]) => count >= 2 && count <= 10)
        .sort((a, b) => a[1] - b[1]);
      
      if (sortedColumns.length > 0) {
        colorColumn = sortedColumns[0][0];
      }
    }
    
    // Choose appropriate scale types based on data characteristics
    // For numerical columns, use linear scales with proper domains
    const xColumn = numericalColumns.length > 0 ? numericalColumns[0] : Object.keys(firstItem)[0];
    const yColumn = numericalColumns.length > 1 ? numericalColumns[1] : (numericalColumns.length > 0 ? numericalColumns[0] : Object.keys(firstItem)[1]);
    
    // Create scales with safe domain calculations
    const xDomain = d3.extent(data, d => +d[xColumn]);
    // Add a small buffer to the domain for better visualization
    const xBuffer = (xDomain[1] - xDomain[0]) * 0.05;
    
    const x = d3.scaleLinear()
      .domain([xDomain[0] - xBuffer, xDomain[1] + xBuffer])
      .range([0, innerWidth])
      .nice();
    
    const yDomain = d3.extent(data, d => +d[yColumn]);
    const yBuffer = (yDomain[1] - yDomain[0]) * 0.05;
    
    const y = d3.scaleLinear()
      .domain([yDomain[0] - yBuffer, yDomain[1] + yBuffer])
      .range([innerHeight, 0])
      .nice();
    
    // Create color scale if we have a suitable categorical column
    let colorScale = null;
    let colorValues = [];
    
    if (colorColumn) {
      colorValues = [...new Set(data.map(d => d[colorColumn]))];
      // Use a categorical color scale with distinct colors
      const colorSchemes = {
        2: d3.schemeSet2,
        3: d3.schemeSet2,
        4: d3.schemeSet2,
        5: d3.schemeSet2,
        6: d3.schemeSet2,
        7: d3.schemeSet2,
        8: d3.schemeSet2,
        9: d3.schemeSet2,
        10: d3.schemeSet2
      };
      
      const colorRange = colorSchemes[Math.min(colorValues.length, 10)] || d3.schemeCategory10;
      colorScale = d3.scaleOrdinal()
        .domain(colorValues)
        .range(colorRange);
    }
    
    // Create and add x-axis with proper formatting
    const xAxis = d3.axisBottom(x)
      .ticks(width > 500 ? 10 : 5)
      .tickPadding(8)
      .tickFormat(d => d3.format(".1f")(d));
    
    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .append("text")
      .attr("class", "axis-label")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "black")
      .attr("text-anchor", "middle")
      .text(xColumn);
    
    // Add horizontal gridlines for better readability
    g.append("g")
      .attr("class", "grid-lines")
      .attr("opacity", 0.1)
      .selectAll("line")
      .data(x.ticks(10))
      .enter()
      .append("line")
      .attr("x1", d => x(d))
      .attr("x2", d => x(d))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "#ccc");
    
    // Create and add y-axis with proper formatting
    const yAxis = d3.axisLeft(y)
      .ticks(height > 400 ? 10 : 5)
      .tickPadding(8)
      .tickFormat(d => d3.format(".1f")(d));
    
    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis)
      .append("text")
      .attr("class", "axis-label")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "black")
      .attr("text-anchor", "middle")
      .text(yColumn);
    
    // Add vertical gridlines
    g.append("g")
      .attr("class", "grid-lines")
      .attr("opacity", 0.1)
      .selectAll("line")
      .data(y.ticks(10))
      .enter()
      .append("line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", d => y(d))
      .attr("y2", d => y(d))
      .attr("stroke", "#ccc");
    
    // Add points with proper color-coding if available
    if (colorScale) {
      // Group data by color for more efficient rendering
      const groupedData = d3.group(data, d => d[colorColumn]);
      
      // Create a group for each color value
      Array.from(groupedData.entries()).forEach(([colorValue, points]) => {
        g.append("g")
          .attr("class", `point-group-${colorValue.replace(/[^a-zA-Z0-9]/g, "")}`)
          .selectAll("circle")
          .data(points)
          .enter()
          .append("circle")
          .attr("cx", d => x(+d[xColumn]))
          .attr("cy", d => y(+d[yColumn]))
          .attr("r", 4)
          .attr("fill", colorScale(colorValue))
          .attr("opacity", 0.7)
          .attr("stroke", "#fff")
          .attr("stroke-width", 0.5);
      });
      
      // Add a legend for color categories
      const legend = svgElement.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - margin.right + 10}, ${margin.top})`);
      
      colorValues.forEach((value, i) => {
        const legendItem = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`);
        
        legendItem.append("rect")
          .attr("width", 12)
          .attr("height", 12)
          .attr("fill", colorScale(value));
        
        legendItem.append("text")
          .attr("x", 20)
          .attr("y", 10)
          .attr("text-anchor", "start")
          .attr("dominant-baseline", "middle")
          .style("font-size", "12px")
          .text(value);
      });
    } else {
      // If no color coding, use a single color for all points
      g.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", d => x(+d[xColumn]))
        .attr("cy", d => y(+d[yColumn]))
        .attr("r", 4)
        .attr("fill", "#4682b4")
        .attr("opacity", 0.7)
        .attr("stroke", "#fff")
        .attr("stroke-width", 0.5);
    }
    
    // Add a title to the visualization
    svgElement.append("text")
      .attr("class", "title")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(`Comparison of ${xColumn} vs ${yColumn}${colorColumn ? ` by ${colorColumn}` : ''}`);
    
    // Add interaction - tooltip for data points
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("position", "absolute")
      .style("background", "rgba(255, 255, 255, 0.9)")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("box-shadow", "0 0 6px rgba(0,0,0,0.3)")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 1000);
    
    g.selectAll("circle")
      .on("mouseover", function(event, d) {
        d3.select(this)
          .attr("r", 6)
          .attr("stroke-width", 1.5);
        
        tooltip.transition()
          .duration(200)
          .style("opacity", 0.9);
        
        let tooltipContent = `
          <strong>${xColumn}:</strong> ${d[xColumn]}<br/>
          <strong>${yColumn}:</strong> ${d[yColumn]}<br/>
        `;
        
        if (colorColumn) {
          tooltipContent += `<strong>${colorColumn}:</strong> ${d[colorColumn]}<br/>`;
        }
        
        tooltip.html(tooltipContent)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        d3.select(this)
          .attr("r", 4)
          .attr("stroke-width", 0.5);
        
        tooltip.transition()
          .duration(500)
          .style("opacity", 0);
      });
    
    // Clean up tooltip when visualization is removed
    return function cleanup() {
      tooltip.remove();
    };
  } catch (error) {
    console.error("Error in visualization:", error);
    
    // Create a fallback visualization with error message
    svgElement.selectAll("*").remove();
    
    const errorG = svgElement.append("g")
      .attr("class", "error-container");
    
    errorG.append("rect")
      .attr("width", svgElement.attr("width"))
      .attr("height", svgElement.attr("height"))
      .attr("fill", "#f8f9fa");
    
    errorG.append("text")
      .attr("x", svgElement.attr("width") / 2)
      .attr("y", svgElement.attr("height") / 2 - 20)
      .attr("text-anchor", "middle")
      .attr("fill", "#d9534f")
      .style("font-weight", "bold")
      .text("Visualization Error");
    
    errorG.append("text")
      .attr("x", svgElement.attr("width") / 2)
      .attr("y", svgElement.attr("height") / 2 + 10)
      .attr("text-anchor", "middle")
      .attr("fill", "#555")
      .text(error.message || "Unknown error occurred");
  }
}
"""
        
        # Build the prompt for the OpenAI API
        prompt = f"""
You are tasked with creating a D3.js visualization to compare two datasets. Your job is to generate JavaScript code that will visualize the data effectively using D3.js version 7.

The visualization will be embedded in a Streamlit app and should follow D3.js best practices, including proper scales, axes, and interactions.

DATA INFORMATION:
- The data is available as an array of objects (JSON)
- Schema information: {json.dumps(columns_info)}
- Sample data: {json.dumps(sample_data, indent=2)}

USER REQUIREMENTS:
{user_input}

REQUIREMENTS:
1. Create a D3.js visualization optimized for comparative analysis
2. Include proper error handling with try/catch blocks
3. Make the visualization responsive to container dimensions
4. Include appropriate axes, legends, and titles
5. Add interactive elements like tooltips
6. Ensure the code is robust against missing or invalid data
7. Follow D3 version 7 conventions

IMPORTANT:
- Your code must start with 'function createVisualization(data, svgElement) {'
- svgElement is a D3 selection of an SVG element, not a raw DOM element
- The data is passed as an array of objects
- Return ONLY JavaScript code, no explanations or markdown
- Avoid using d3.json() or external data loading - the data is already provided as 'data'
- Focus on making scales robust by handling edge cases
- Ensure the code handles all potential errors gracefully

Use this template as a starting point, but customize it based on the user requirements:
{visualization_template}

RESPONSE FORMAT:
Only return the JavaScript function as plain text with no backticks or markdown.
"""
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18"),
            messages=[
                {"role": "system", "content": "You are a D3.js expert who creates robust data visualizations with excellent error handling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content.strip()
        
        # Clean up the response to ensure it's valid JavaScript
        d3_code = clean_d3_response(response_content)
        
        # Log successful generation
        logger.info(f"Generated D3 code with user input, code length: {len(d3_code)} characters")
        
        return d3_code
        
    except Exception as e:
        logger.error(f"Error in generate_d3_code: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Provide a fallback visualization if the API call fails
        return """
        function createVisualization(data, svgElement) {
          // Fallback visualization if the API call fails
          try {
            // Set up dimensions and margins
            const width = parseInt(svgElement.attr("width")) || 800;
            const height = parseInt(svgElement.attr("height")) || 500;
            const margin = {top: 50, right: 50, bottom: 50, left: 50};
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;
            
            // Create a group for the visualization
            const g = svgElement.append("g")
              .attr("transform", `translate(${margin.left},${margin.top})`);
              
            // Add an error message
            g.append("text")
              .attr("x", innerWidth / 2)
              .attr("y", innerHeight / 2 - 20)
              .attr("text-anchor", "middle")
              .style("font-size", "16px")
              .style("font-weight", "bold")
              .text("Error generating visualization");
              
            g.append("text")
              .attr("x", innerWidth / 2)
              .attr("y", innerHeight / 2 + 10)
              .attr("text-anchor", "middle")
              .style("font-size", "14px")
              .text("Please try again or modify your request");
              
            // Display some basic data info if available
            if (data && Array.isArray(data) && data.length > 0) {
              const availableColumns = Object.keys(data[0]);
              
              g.append("text")
                .attr("x", innerWidth / 2)
                .attr("y", innerHeight / 2 + 40)
                .attr("text-anchor", "middle")
                .style("font-size", "12px")
                .text(`Available columns: ${availableColumns.join(", ")}`);
            }
          } catch (error) {
            console.error("Error in fallback visualization:", error);
            
            // Absolute fallback
            svgElement.append("text")
              .attr("x", 100)
              .attr("y", 100)
              .style("fill", "red")
              .text("Unable to render visualization");
          }
        }
        """

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
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>D3 Visualization</title>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    #visualization {{
                        width: 100%;
                        height: {VISUALIZATION_HEIGHT}px;
                        margin: 0 auto;
                        background-color: #ffffff;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                        overflow: hidden;
                        position: relative;
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
                    
                    .fallback-viz {{
                        width: 100%;
                        height: 100%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                        background-color: #f8f9fa;
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
                                // Use default domain if missing or invalid
                                const safeDomain = (Array.isArray(domain) && domain.length === 2 && 
                                                   !isNaN(domain[0]) && !isNaN(domain[1])) 
                                    ? domain 
                                    : [0, 100];
                                
                                // Use default range if missing or invalid
                                const safeRange = (Array.isArray(range) && range.length === 2 &&
                                                  !isNaN(range[0]) && !isNaN(range[1]))
                                    ? range
                                    : [0, 500];
                                    
                                // Create the scale with domain padding for better visualization
                                const paddedDomain = [
                                    safeDomain[0] - Math.abs(safeDomain[0] * 0.05),
                                    safeDomain[1] + Math.abs(safeDomain[1] * 0.05)
                                ];
                                    
                                return d3.scaleLinear().domain(paddedDomain).range(safeRange).nice();
                            }} catch (e) {{
                                console.warn("Error creating scale:", e);
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
                                
                                // Add special handling based on scale type
                                let scale;
                                if (scaleName === 'scaleLinear' || scaleName === 'scaleLog' || 
                                    scaleName === 'scalePow' || scaleName === 'scaleTime') {{
                                    // For continuous scales, ensure proper domain and range
                                    scale = scaleFunc();
                                    
                                    // Handle edge cases for certain scale types
                                    if (scaleName === 'scaleLog') {{
                                        // Ensure domain doesn't include zero or negative values for log scales
                                        if (Array.isArray(domain) && (domain[0] <= 0 || domain[1] <= 0)) {{
                                            console.warn("Log scale domain can't include zero or negative values, using fallback domain");
                                            domain = [0.1, 100];
                                        }}
                                    }}
                                    
                                    // Add padding for continuous domains for better visualization
                                    if (Array.isArray(domain) && domain.length === 2 && 
                                        !isNaN(domain[0]) && !isNaN(domain[1])) {{
                                        const paddingAmount = Math.abs(domain[1] - domain[0]) * 0.05;
                                        const paddedDomain = [domain[0] - paddingAmount, domain[1] + paddingAmount];
                                        scale.domain(paddedDomain);
                                    }} else if (Array.isArray(domain)) {{
                                        scale.domain(domain);
                                    }}
                                    
                                    if (Array.isArray(range)) scale.range(range);
                                    scale.nice && scale.nice();
                                }} else if (scaleName === 'scaleOrdinal' || scaleName === 'scaleBand') {{
                                    // For ordinal scales, handle domain and range differently
                                    scale = scaleFunc();
                                    if (Array.isArray(domain)) scale.domain(domain);
                                    if (Array.isArray(range)) {{
                                        if (scaleName === 'scaleBand') {{
                                            scale.range(range).padding(0.1);
                                        }} else {{
                                            scale.range(range);
                                        }}
                                    }}
                                }} else {{
                                    // For other scale types
                                    scale = scaleFunc();
                                    if (Array.isArray(domain)) scale.domain(domain);
                                    if (Array.isArray(range)) scale.range(range);
                                }}
                                
                                // Test if scale has proper methods before using them
                                if (typeof scale.domain !== 'function' || typeof scale.range !== 'function') {{
                                    console.warn(`Created scale doesn't have proper methods, using linear`);
                                    return this.createLinearScale(domain, range);
                                }}
                                
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
                                    
                                    // Test the scale with sample values
                                    const testInput = (scale.domain()[1] + scale.domain()[0]) / 2;
                                    const testOutput = scale(testInput);
                                    
                                    // If the output is NaN or undefined, the scale isn't working properly
                                    if (isNaN(testOutput) && typeof testOutput !== 'string') {{
                                        console.warn("Scale produces NaN output, creating fallback scale");
                                        return this.createLinearScale(defaultDomain, defaultRange);
                                    }}
                                    
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
                                
                                // Set tick count safely and format ticks
                                try {{
                                    if (Number.isInteger(tickCount) && tickCount > 0) {{
                                        axis.ticks(tickCount);
                                    }}
                                    
                                    // Add better tick formatting based on scale type
                                    if (typeof scale.domain()[0] === 'number') {{
                                        // For numerical scales, use nice number formatting
                                        axis.tickFormat(d => {{
                                            // Use appropriate precision based on the domain range
                                            const domain = scale.domain();
                                            const range = Math.abs(domain[1] - domain[0]);
                                            if (range <= 1) return d3.format(".2f")(d);
                                            if (range <= 10) return d3.format(".1f")(d);
                                            return d3.format(",.0f")(d);
                                        }});
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
                            return d[property] !== undefined ? d[property] : defaultValue;
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
                        
                        // Create a color scale with better defaults
                        createColorScale: function(domain, range) {{
                            try {{
                                // If no domain provided, create a default one
                                const safeDomain = Array.isArray(domain) && domain.length > 0 
                                    ? domain 
                                    : ['A', 'B'];
                                
                                // If no range provided, use a colorful default
                                const safeRange = Array.isArray(range) && range.length > 0
                                    ? range
                                    : d3.schemeCategory10;
                                
                                return d3.scaleOrdinal()
                                    .domain(safeDomain)
                                    .range(safeRange);
                            }} catch (e) {{
                                console.warn("Error creating color scale:", e);
                                return d3.scaleOrdinal(['A', 'B'], ['#1f77b4', '#ff7f0e']);
                            }}
                        }},
                        
                        // Helper to create properly formatted axes with labels
                        createAxesWithLabels: function(svgElement, xScale, yScale, options = {}) {{
                            const defaults = {{
                                width: 600,
                                height: 400,
                                margin: {{top: 40, right: 40, bottom: 40, left: 40}},
                                xLabel: "X Axis",
                                yLabel: "Y Axis",
                                ticksX: 5,
                                ticksY: 5
                            }};
                            
                            // Merge options with defaults
                            const config = {...defaults, ...options};
                            if (options.margin) {{
                                config.margin = {...defaults.margin, ...options.margin};
                            }}
                            
                            try {{
                                // Verify scales
                                const safeXScale = this.verifyScale(xScale);
                                const safeYScale = this.verifyScale(yScale);
                                
                                // Create group for axes if not already present
                                let g = svgElement.select("g.axes-container");
                                if (g.empty()) {{
                                    g = svgElement.append("g")
                                        .attr("class", "axes-container")
                                        .attr("transform", `translate(${{config.margin.left}},${{config.margin.top}})`);
                                }}
                                
                                // Calculate inner dimensions
                                const innerWidth = config.width - config.margin.left - config.margin.right;
                                const innerHeight = config.height - config.margin.top - config.margin.bottom;
                                
                                // Create and add x-axis
                                const xAxis = this.createAxis(safeXScale, 'bottom', config.ticksX);
                                const xAxisG = g.append("g")
                                    .attr("class", "x-axis")
                                    .attr("transform", `translate(0,${{innerHeight}})`)
                                    .call(xAxis);
                                
                                // Add x-axis label
                                xAxisG.append("text")
                                    .attr("class", "x-axis-label")
                                    .attr("x", innerWidth / 2)
                                    .attr("y", 35)
                                    .attr("fill", "black")
                                    .attr("text-anchor", "middle")
                                    .text(config.xLabel);
                                
                                // Create and add y-axis
                                const yAxis = this.createAxis(safeYScale, 'left', config.ticksY);
                                const yAxisG = g.append("g")
                                    .attr("class", "y-axis")
                                    .call(yAxis);
                                
                                // Add y-axis label
                                yAxisG.append("text")
                                    .attr("class", "y-axis-label")
                                    .attr("transform", "rotate(-90)")
                                    .attr("x", -innerHeight / 2)
                                    .attr("y", -35)
                                    .attr("fill", "black")
                                    .attr("text-anchor", "middle")
                                    .text(config.yLabel);
                                
                                return {{
                                    xAxis: xAxisG,
                                    yAxis: yAxisG,
                                    innerWidth: innerWidth,
                                    innerHeight: innerHeight
                                }};
                            }} catch (e) {{
                                console.warn("Error creating axes with labels:", e);
                                return null;
                            }}
                        }}
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
                            let data = {json.dumps(st.session_state.json_data)};
                            
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
            error_msg = f"Error displaying visualization. Please check the browser console for details. Error: {str(e)}"
            logger.error(f"Error in display_visualization: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Display error message if an error occurred
    if error_occurred:
        if placeholder is not None:
            with placeholder:
                st.error(error_msg)
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
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18"),
            messages=[
                {"role": "system", "content": "You are a D3.js expert who creates robust data visualizations with excellent error handling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3500
        )
        
        # Extract the code from the response
        new_code = response.choices[0].message.content.strip()
        
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

if __name__ == "__main__":
    main()
