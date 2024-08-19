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
import streamlit.components.v1 as components
import base64

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
    
    This function attempts to get the API key from Streamlit secrets or environment variables.
    If not found, it uses the value entered in the sidebar input.
    
    Returns:
        Optional[str]: The API key if found or entered, None otherwise.
    """
    return st.secrets.get("API_KEY") or os.getenv("API_KEY") or st.session_state.get('api_key')

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
        
        logger.info(f"File 1 shape: {df1.shape}, File 2 shape: {df2.shape}")
        
        # Add 'Source' column to identify the origin of each row
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'
        
        # Merge the two DataFrames
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        
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
        logger.info(f"Final DataFrame shape: {merged_df.shape}")
        logger.info(f"Columns: {merged_df.columns.tolist()}")
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

def generate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "", selected_model: str = "GPT-4o (High-intelligence flagship)") -> str:
    """
    Generate D3.js code using OpenAI API with emphasis on comparison and readability.
    
    This function constructs a prompt for the OpenAI API, including data schema and sample,
    and generates D3.js code based on the input DataFrame and user requirements.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
        selected_model (str, optional): The selected model for code generation.
    
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
    
    d3_code = ""

    base_prompt = f"""
    # D3.js Code Generation Task

    Generate ONLY D3.js version 7 code for a clear, readable, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

    Critical Requirements for D3.js Visualization:
    1. Create a function named createVisualization(data, svgElement)
    2. Set up an SVG canvas with margins, width, and height as specified - const svgWidth = 1200, svgHeight = 700
    3. Implement a color palette using d3.scaleOrdinal(d3.schemePastel1)
    4. Add a subtle background rectangle with rounded corners
    5. Create scales for x-axis, y-axis, and color.    
        - X-axis: d3.scaleBand()
        - Y-axis: d3.scaleLinear()
    6. Implement an animated area chart with gradient fill where applicable.
    7. Add an animated line chart on top of the area chart 
        - Use d3.line() to define the line shape
        - Animate the line using d3.transition() and attrTween('d', function(d))
        - Implement path interpolation with d3.interpolate() for smooth animation
    8. Add chart title and axis labels using d3.text()
    9. Create interactive axes with proper formatting and rotated labels (45 degrees) if needed.
    10. Implement interactive data points with hover effects and smooth transitions.
    11. Design an informative tooltip that appears on hover and can be locked on click.
    12. Create a dynamic and interactive legend that highlights data on hover.
    13. Implement zooming and panning functionality.
    14. Add a crosshair effect for precise data reading.
    15. Implement a brush for range selection.
    16. Ensure smooth color transitions and micro-interactions.
    17. Include error checking for invalid data formats and handle missing data.
    18. Optimize performance using efficient D3 methods and requestAnimationFrame.
    19. Ensure accessibility with ARIA labels and d3-textwrap for long labels.
    20. Implement responsive design that adjusts to window resizing.
    21. Remember to comply with the user's request intelligently, updating existing code if it's an update request, or creating new code if it's a new visualization request. Always return the complete, updated code.
    22. The nature of visualization is comparative. So the user will be comparing multiple data sets. So the visualization must explicitly show the comparison and highlight the differences.
    23. You must understand how exactly the source data is different from each other and show the differences in the visualization intelligently by pointing out the differentiating factors.

    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    User Request:
    {user_input}

    IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
    """
    
    prompt = base_prompt
    
    try:
        if "OpenAI" in selected_model:
            client = OpenAI(api_key=api_key)
            model_name = "gpt-4-1106-preview" if "GPT-4" in selected_model else "gpt-3.5-turbo"
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a D3.js expert specializing in creating clear, readable, and comparative visualizations."},
                    {"role": "user", "content": prompt}
                ]
            )
            d3_code = response.choices[0].message.content
        elif "Claude" in selected_model:
            # Implement Anthropic API call here
            # You'll need to use the appropriate Anthropic client library or API endpoint
            # For example:
            # from anthropic import Anthropic
            # client = Anthropic(api_key=api_key)
            # response = client.completions.create(
            #     model="claude-3-sonnet-20240229",
            #     prompt=f"Human: {prompt}\n\nAssistant:",
            #     max_tokens_to_sample=1000
            # )
            # d3_code = response.completion
            pass
        elif "Gemini" in selected_model:
            # Implement Google AI (Gemini) API call here
            # You'll need to use the appropriate Google AI client library or API endpoint
            # For example:
            # from google.generativeai import GenerativeModel
            # model = GenerativeModel('gemini-pro')
            # response = model.generate_content(prompt)
            # d3_code = response.text
            pass
        else:
            raise ValueError(f"Unsupported model: {selected_model}")

        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")
        
        cleaned_d3_code = clean_d3_response(d3_code)
        return cleaned_d3_code
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
    
    This function removes markdown formatting, code block delimiters,
    non-JavaScript lines, and ensures the code starts with the 
    createVisualization function.
    
    Args:
        response (str): The raw response from the LLM.
    
    Returns:
        str: Cleaned D3.js code.
    """
    # Remove any potential markdown code blocks
    response = re.sub(r'```(?:javascript)?\n?', '', response)
    response = response.replace('```', '')
    
    # Remove any lines that don't look like JavaScript
    clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    # Ensure the code starts with the createVisualization function
    if not any(line.strip().startswith('function createVisualization') for line in clean_lines):
        clean_lines.insert(0, 'function createVisualization(data, svgElement) {')
        clean_lines.append('}')
    
    return '\n'.join(clean_lines)

def display_visualization(d3_code: str, encoded_data: str) -> None:
    """
    Prepare the HTML content for the D3.js visualization and display it using an iframe.
    
    Args:
        d3_code (str): The D3.js code to be executed.
        encoded_data (str): URL-encoded JSON data for the visualization.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>D3 Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://d3js.org/d3-scale-chromatic.v1.min.js"></script>
        <style>
            #visualization {{
                width: 100%;
                height: 100vh;
                overflow: auto;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <script>
            try {{
                const data = JSON.parse(decodeURIComponent("{encoded_data}"));
                console.log("Parsed data:", data);
                const svgElement = d3.select("#visualization")
                    .append("svg")
                    .attr("width", "100%")
                    .attr("height", "100%")
                    .attr("viewBox", "0 0 1000 600")
                    .attr("preserveAspectRatio", "xMidYMid meet")
                    .node();
                
                {d3_code}
                
                createVisualization(data, svgElement);
                console.log("Visualization created successfully");
            }} catch (error) {{
                console.error("Error in visualization:", error);
                document.getElementById("visualization").innerHTML = "Error: " + error.message;
            }}
        </script>
    </body>
    </html>
    """
    
    # Create a data URI for the HTML content
    html_uri = f"data:text/html;base64,{base64.b64encode(html_content.encode()).decode()}"
    
    # Display the visualization using st.components.v1.iframe
    st.components.v1.iframe(html_uri, height=600, scrolling=True)

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

    # Add model selection dropdown in the sidebar
    model_options = {
        "OpenAI": [
            "GPT-4o (High-intelligence flagship)",
            "GPT-4o mini (Affordable and intelligent)",
            "GPT-4 Turbo",
            "GPT-4",
            "GPT-3.5 Turbo (Fast and inexpensive)"
        ],
        "Anthropic": [
            "Claude 3.5 Sonnet (Most intelligent)",
            "Claude 3 Haiku (Fast and cost-effective)",
            "Claude 3 Sonnet (Balanced)",
            "Claude 3 Opus (Excels at writing and complex tasks)"
        ],
        "Google": [
            "Gemini Pro"
        ]
    }
    
    selected_provider = st.sidebar.selectbox("Select AI Provider", options=list(model_options.keys()))
    selected_model = st.sidebar.selectbox("Select Model", options=model_options[selected_provider])

    # Update the API key input to be generic
    api_key = st.sidebar.text_input("Enter your API Key", type="password")
    if api_key:
        st.sidebar.warning("It's recommended to use environment variables or Streamlit secrets for API keys.")

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
                st.write(f"DataFrame shape: {merged_df.shape}")
                st.write(f"Columns: {', '.join(merged_df.columns)}")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during data preprocessing: {str(e)}")
                logger.error(f"Preprocessing error: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return
    else:
        if 'preprocessed_df' in st.session_state:
            with st.expander("Preview of preprocessed data"):
                st.dataframe(st.session_state.preprocessed_df.head())
                st.write(f"Total rows: {len(st.session_state.preprocessed_df)}")
                st.write(f"Columns: {', '.join(st.session_state.preprocessed_df.columns)}")

        st.subheader("Generate Visualization")
        user_query = st.text_area("Enter your visualization request:", height=100)
        
        if st.button("Generate Visualization"):
            if user_query and api_key:
                try:
                    with st.spinner("Generating visualization..."):
                        d3_code = generate_d3_code(st.session_state.preprocessed_df, api_key, user_query, selected_model)
                        cleaned_d3_code = clean_d3_response(d3_code)
                    st.session_state.current_viz = cleaned_d3_code
                    st.session_state.workflow_history.append({
                        "version": len(st.session_state.workflow_history) + 1,
                        "request": user_query,
                        "code": cleaned_d3_code
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred while generating the visualization: {str(e)}")
                    logger.error(f"Visualization generation error: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                if not user_query:
                    st.warning("Please enter a visualization request.")
                if not api_key:
                    st.warning("Please enter a valid API key.")

        if 'current_viz' in st.session_state:
            st.subheader("Current Visualization")
            try:
                with st.spinner("Preparing visualization..."):
                    html_content = display_visualization(st.session_state.current_viz)
                    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))
                    iframe_url = f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}"
                    components.html(html_content, height=600, scrolling=True)
                    st.components.v1.iframe(iframe_url, width=1000, height=600, scrolling=True)
            except Exception as e:
                st.error(f"An error occurred while displaying the visualization: {str(e)}")
                st.error("Please check the browser console for more details.")
                st.code(st.session_state.current_viz, language="javascript")

            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area("D3.js Code", value=st.session_state.current_viz, height=300, key="code_editor")
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    edit_enabled = st.toggle("Edit", key="edit_toggle")
                with col2:
                    if st.button("Execute Code"):
                        if edit_enabled:
                            if validate_d3_code(code_editor):
                                cleaned_code = clean_d3_response(code_editor)
                                st.session_state.current_viz = cleaned_code
                                st.session_state.workflow_history.append({
                                    "version": len(st.session_state.workflow_history) + 1,
                                    "request": "Manual code edit",
                                    "code": cleaned_code
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

    # Add this at the end of the main function
    with st.expander("Debug Information"):
        st.write("Session State:")
        for key, value in st.session_state.items():
            if key != 'preprocessed_df':  # Avoid displaying large dataframes
                st.write(f"{key}: {value}")
        if 'preprocessed_df' in st.session_state:
            st.write("Preprocessed DataFrame Info:")
            st.write(st.session_state.preprocessed_df.info())

if __name__ == "__main__":
    main()
