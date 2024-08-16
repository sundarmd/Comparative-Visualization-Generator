import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json
import logging
import traceback
from typing import Optional, Dict, List
import re
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define MAX_WORKFLOW_HISTORY constant
MAX_WORKFLOW_HISTORY = 10

# Initialize session state
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []  # Stores the history of visualization changes
if 'current_viz' not in st.session_state:
    st.session_state.current_viz = None  # Stores the current D3.js visualization code
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None  # Stores the preprocessed DataFrame
if 'update_viz' not in st.session_state:
    st.session_state.update_viz = False  # Flag to trigger visualization update
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Stores the chat history

def get_api_key() -> Optional[str]:
    """
    Securely retrieve the API key.
    
    This function attempts to get the OpenAI API key from Streamlit secrets or environment variables.
    If not found, it prompts the user to enter the key via a sidebar input.
    
    Returns:
        Optional[str]: The API key if found or entered, None otherwise.
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.sidebar.warning("It's recommended to use environment variables or Streamlit secrets for API keys.")
    return api_key

def test_api_key(api_key: str) -> bool:
    """
    Test if the provided API key is valid.
    
    This function attempts to list OpenAI models using the provided API key.
    If successful, the key is considered valid.
    
    Args:
        api_key (str): The OpenAI API key to test.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
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

def validate_d3_code(code: str) -> bool:
    """
    Perform basic validation on the generated D3 code.
    
    This function checks for the presence of key D3.js elements and syntax.
    
    Args:
        code (str): The D3.js code to validate.
    
    Returns:
        bool: True if the code passes basic validation, False otherwise.
    """
    # Check if the code defines the createVisualization function
    if not re.search(r'function\s+createVisualization\s*\(data,\s*svgElement\)\s*{', code):
        return False
    
    # Check for basic D3 v7 method calls
    d3_methods = ['d3.select', 'd3.scaleLinear', 'd3.axisBottom', 'd3.axisLeft']
    if not any(method in code for method in d3_methods):
        return False
    
    # Check for balanced braces
    if code.count('{') != code.count('}'):
        return False
    
    return True

def generate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    """
    Generate D3.js code using OpenAI API with emphasis on intelligence and insights.
    
    This function constructs a comprehensive prompt for the OpenAI API, including
    system message, data schema, sample data, and user input.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
    
    Returns:
        str: Generated D3.js code.
    
    Raises:
        ValueError: If generated D3 code is empty.
        Exception: For any errors during API call or code generation.
    """
    logger.info("Starting D3 code generation")
    data_sample = df.head(50).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
    
    client = OpenAI(api_key=api_key)
    
    system_message = """You are an elite D3.js expert specializing in creating sophisticated, insightful, and intelligent visualizations. 
    Your task is to generate D3.js version 7 code that not only visualizes data but also reveals deep insights and patterns. 
    Your visualizations should be clear, readable, comparative, and demonstrate advanced analytical thinking."""

    prompt = f"""
    # Intelligent D3.js Visualization Generation Task

    Generate ONLY D3.js version 7 code for a sophisticated, insightful, and intelligent visualization. Do not include any explanations, comments, or markdown formatting.

    Critical Requirements:
    1. Create a function named createVisualization(data, svgElement)
    2. Implement an advanced visualization that reveals deep insights:
       - Use intelligent color schemes that enhance data comprehension
       - Incorporate multiple layers of information (e.g., combine chart types)
       - Highlight key trends, outliers, or patterns automatically
    3. Implement smart comparative features:
       - Use advanced techniques like small multiples or parallel coordinates for multi-dimensional comparisons
       - Incorporate statistical measures (e.g., mean, median, quartiles) to add context
    4. Solve data complexity issues:
       - Implement intelligent aggregation or sampling for large datasets
       - Use advanced labeling techniques (e.g., smart label placement algorithms)
    5. Add interactive elements that reveal deeper insights:
       - Implement zoom and pan for detailed exploration
       - Add click-through capabilities to drill down into data points
    6. Ensure the visualization is self-explanatory:
       - Add smart annotations that highlight key insights
       - Implement an intelligent legend that adapts to the data
    7. Optimize for performance:
       - Use efficient D3 methods for large dataset handling
       - Implement smart data binding and update patterns
    8. If applicable, address the following user request intelligently:
       {user_input}

    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    Current Visualization Code (if any):
    ```javascript
    {st.session_state.get('current_viz', 'No current visualization')}
    ```

    IMPORTANT: Your entire response must be valid, sophisticated D3.js code that can be executed directly. The code should demonstrate advanced D3.js techniques and intelligent data visualization practices. Do not include any text before or after the code.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        
        d3_code = response.choices[0].message.content
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")
        
        return d3_code
    except Exception as e:
        logger.error(f"Error generating D3 code: {str(e)}")
        return generate_fallback_visualization()

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
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_attempts):
        if validate_d3_code(initial_code):
            return initial_code
        
        refinement_prompt = f"""
        The following D3 code needs refinement to be valid:
        
        {initial_code}
        
        Please provide a corrected version that:
        1. Defines a createVisualization(data, svgElement) function
        2. Uses only D3.js version 7 syntax
        3. Creates a valid visualization
        
        Return ONLY the corrected D3 code without any explanations or comments.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Provide only valid D3 code."},
                {"role": "user", "content": refinement_prompt}
            ]
        )
        
        initial_code = clean_d3_response(response.choices[0].message.content)
    
    # If we've exhausted our attempts, return the last attempt
    logger.warning("Failed to generate valid D3 code after maximum attempts")
    return initial_code

def clean_d3_response(response: str) -> str:
    """
    Clean the LLM response to ensure it only contains D3 code.
    
    This function removes markdown formatting, non-JavaScript lines,
    and ensures the code starts with the createVisualization function.
    
    Args:
        response (str): The raw response from the LLM.
    
    Returns:
        str: Cleaned D3.js code.
    """
    # Remove any potential markdown code blocks
    response = response.replace("```javascript", "").replace("```", "")
    
    # Remove any lines that don't look like JavaScript
    clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    # Ensure the code starts with the createVisualization function
    if not any(line.strip().startswith('function createVisualization') for line in clean_lines):
        clean_lines.insert(0, 'function createVisualization(data, svgElement) {')
        clean_lines.append('}')
    
    return '\n'.join(clean_lines)

def display_visualization(d3_code: str):
    """
    Display the D3.js visualization using an iframe with improved error handling and flexibility.
    
    Args:
        d3_code (str): The D3.js code to be executed.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            #visualization {{
                width: 100%;
                height: 100vh;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <script>
            try {{
                {d3_code}
                // Create the SVG element with responsive dimensions
                const svgElement = d3.select("#visualization")
                    .append("svg")
                    .attr("width", "100%")
                    .attr("height", "100%")
                    .attr("viewBox", "0 0 800 500")
                    .attr("preserveAspectRatio", "xMidYMid meet")
                    .node();
                
                // Get the data from the parent window
                const vizData = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
                
                // Call the createVisualization function
                createVisualization(vizData, svgElement);
            }} catch (error) {{
                console.error("Error in D3 visualization:", error);
                document.body.innerHTML = `<p>Error rendering visualization: ${{error.message}}</p>`;
            }}
        </script>
    </body>
    </html>
    """
    
    # Encode the data to pass it to the iframe
    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))
    
    # Display the iframe with the encoded data in the URL hash
    st.components.v1.iframe(
        src=f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}", 
        width="800",  # Changed from "100%" to a fixed width
        height=600,
        scrolling=True
    )

def generate_fallback_visualization() -> str:
    """
    Generate a fallback visualization if the LLM fails.
    
    This function creates a simple bar chart using D3.js as a fallback
    when the main visualization generation process fails.
    
    Returns:
        str: D3.js code for a simple bar chart visualization.
    """
    logger.info("Generating fallback visualization")
    
    fallback_code = """
    function createVisualization(data, svgElement) {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const width = 800 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;
        
        svgElement.attr("width", width + margin.left + margin.right)
                   .attr("height", height + margin.top + margin.bottom);
        
        const svg = svgElement.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // Assuming the first column is for x-axis and second for y-axis
        const xKey = Object.keys(data[0])[0];
        const yKey = Object.keys(data[0])[1];

        const xScale = d3.scaleBand()
            .domain(data.map(d => d[xKey]))
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => +d[yKey])])
            .range([height, 0]);

        svg.selectAll("rect")
            .data(data)
            .join("rect")
            .attr("x", d => xScale(d[xKey]))
            .attr("y", d => yScale(+d[yKey]))
            .attr("width", xScale.bandwidth())
            .attr("height", d => height - yScale(+d[yKey]))
            .attr("fill", "steelblue");

        svg.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(xScale));

        svg.append("g")
            .call(d3.axisLeft(yScale));

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + margin.top + 20)
            .attr("text-anchor", "middle")
            .text(xKey);

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -margin.left + 20)
            .attr("text-anchor", "middle")
            .text(yKey);
    }
    """
    
    logger.info("Fallback visualization generated successfully")
    return fallback_code

def generate_and_validate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    """Generate, validate, and if necessary, refine D3 code."""
    initial_code = generate_d3_code(df, api_key, user_input)
    cleaned_code = clean_d3_response(initial_code)
    
    if validate_d3_code(cleaned_code):
        return cleaned_code
    else:
        return refine_d3_code(cleaned_code, api_key)

def main():
    st.set_page_config(page_title="🎨 Comparative Visualization Generator", page_icon="✨", layout="wide")
    st.title("🎨 Comparative Visualization Generator")

    api_key = get_api_key()

    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False

    if not st.session_state.data_uploaded:
        st.header("Upload CSV Files")
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Upload first CSV file", type="csv")
        with col2:
            file2 = st.file_uploader("Upload second CSV file", type="csv")

        if file1 and file2:
            try:
                with st.spinner("Preprocessing data..."):
                    merged_df = preprocess_data(file1, file2)
                st.session_state.preprocessed_df = merged_df
                st.session_state.data_uploaded = True
                st.success("Data uploaded and preprocessed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during data preprocessing: {str(e)}")
                return
    else:
        if 'preprocessed_df' in st.session_state:
            with st.expander("Preview of preprocessed data"):
                st.dataframe(st.session_state.preprocessed_df.head())

        st.subheader("Generate Visualization")
        user_query = st.text_area("Enter your visualization request:", height=100)
        
        if st.button("Generate Visualization"):
            if user_query:
                with st.spinner("Generating visualization..."):
                    d3_code = generate_d3_code(st.session_state.preprocessed_df, api_key, user_query)
                st.session_state.current_viz = d3_code
                st.session_state.workflow_history.append({
                    "version": len(st.session_state.workflow_history) + 1,
                    "request": user_query,
                    "code": d3_code
                })
                st.rerun()
            else:
                st.warning("Please enter a visualization request.")

        if 'current_viz' in st.session_state:
            st.subheader("Current Visualization")
            with st.spinner("Preparing visualization..."):
                display_visualization(st.session_state.current_viz)

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
                                if len(st.session_state.workflow_history) > MAX_WORKFLOW_HISTORY:
                                    st.session_state.workflow_history.pop(0)
                                st.rerun()
                            else:
                                st.error("Invalid D3.js code. Please check your code and try again.")
                        else:
                            st.warning("Enable 'Edit' to make changes.")
                with col3:
                    if st.button("Copy Code"):
                        st.write("Code copied to clipboard!")
                        st.write(f'<textarea style="position: absolute; left: -9999px;">{code_editor}</textarea>', unsafe_allow_html=True)
                        st.write('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)

            with st.expander("Workflow History"):
                for i, step in enumerate(st.session_state.workflow_history):
                    st.subheader(f"Step {i+1}")
                    st.write(f"Request: {step['request']}")
                    if st.button(f"Revert to Step {i+1}"):
                        st.session_state.current_viz = step['code']
                        st.rerun()

        if st.button("End Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session ended. You can start a new session by uploading new files.")
            st.rerun()

if __name__ == "__main__":
    main()
