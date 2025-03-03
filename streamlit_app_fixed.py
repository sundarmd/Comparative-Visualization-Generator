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
    Display the D3.js visualization in the Streamlit app using components.html.
    
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
                                
                            return d3.scaleLinear().domain(safeDomain).range(safeRange);
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
        error_msg = f"Error displaying visualization. Please check the browser console for details. Error: {str(e)}"
        logger.error(f"Error in display_visualization: {str(e)}")
        logger.error(traceback.format_exc())
        
        if placeholder is not None:
            with placeholder:
                st.error(error_msg)
        else:
            st.error(error_msg)


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
