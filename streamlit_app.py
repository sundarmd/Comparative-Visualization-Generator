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
    Generate D3.js code using OpenAI API with emphasis on comparison and readability.
    
    This function constructs a prompt for the OpenAI API, including data schema and sample,
    and generates D3.js code based on the input DataFrame and user requirements.
    
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
    
    d3_code = ""

    client = OpenAI(api_key=api_key)
    
    base_prompt = f"""
    # D3.js Code Generation Task

    Generate ONLY D3.js version 7 code for a clear, readable, and comparative visualization. Do not include any explanations, comments, or markdown formatting.


    Critical Requirements for D3.js Visualization:

    1. Create a function named createVisualization(data, svgElement)

    2. Set up an advanced SVG canvas:
    - Define margins using const margin = [top: 40, right: 100, bottom: 80, left: 100]
    - Set width and height: const width = 1200 - margin.left - margin.right, height = 700 - margin.top - margin.bottom
    - Create an SVG element with d3.select(svgElement).attr("viewBox", `0 0 $[width + margin.left + margin.right] $[height + margin.top + margin.bottom]`)
    - Add a subtle background with svg.append("rect").attr("width", width).attr("height", height).attr("fill", "#f8f9fa").attr("rx", 10).attr("ry", 10)

    3. Implement sophisticated scales:
    - X-axis: const xScale = d3.scaleBand().domain(data.map(d => d.x)).range([0, width]).padding(0.1)
    - Y-axis: const yScale = d3.scaleLinear().domain([0, d3.max(data, d => d.y) * 1.1]).range([height, 0])
    - Color: const colorScale = d3.scaleOrdinal(d3.schemePastel1)

    4. Create an animated area and line chart:
    - Define area: const area = d3.area().x(d => xScale(d.x) + xScale.bandwidth() / 2).y0(height).y1(d => yScale(d.y)).curve(d3.curveMonotoneX)
    - Add gradient: Use svg.append("defs").append("linearGradient") for a beautiful gradient fill
    - Animate line: Use path.attr("stroke-dasharray", totalLength).attr("stroke-dashoffset", totalLength).transition().duration(2000).attr("stroke-dashoffset", 0)

    5. Add interactive axes with proper formatting:
    - X-axis: svg.append("g").attr("class", "x axis").attr("transform", `translate(0,$[height])`).call(d3.axisBottom(xScale))
    - Y-axis: svg.append("g").attr("class", "y axis").call(d3.axisLeft(yScale).tickFormat(d3.format(',.1f')).ticks(10))
    - Rotate x-axis labels if needed: .selectAll("text").style("text-anchor", "end").attr("dx", "-.8em").attr("dy", ".15em").attr("transform", "rotate(-45)")

    6. Implement highly interactive data points:
    - Create dots: svg.selectAll(".dot").data(data).enter().append("circle")
    - Add hover effects: .on("mouseover", function(event, d) [ d3.select(this).transition().duration(300).attr("r", 8) ])
    - Implement smooth transitions: .transition().delay((d, i) => i * 50).duration(500).attr("r", 5)

    7. Design an informative and interactive tooltip:
    - Create tooltip: const tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0)
    - Show on hover: function showTooltip(event, d) [ tooltip.transition().duration(200).style("opacity", .9).html(`...`).style("left", `$[event.pageX]px`).style("top", `$[event.pageY]px`) ]

    8. Create a dynamic and interactive legend:
    - Legend group: const legend = svg.append("g").attr("class", "legend").attr("transform", `translate($[width + 20], 0)`)
    - Legend items: legend.selectAll(".legend-item").data(colorScale.domain()).enter().append("g")
    - Add interactivity: .on("mouseover", highlightCategory).on("mouseout", resetHighlight)

    9. Implement advanced responsive design:
    - Create resize function: function resize() [ ... ]
    - Update all elements: xScale.range([0, newWidth]), yScale.range([height, 0]), svg.select(".x.axis").call(xAxis), ...
    - Attach to window: d3.select(window).on("resize", resize)

    10. Enhance visual appeal with animations and transitions:
        - Use d3.transition() for smooth updates: svg.selectAll(".dot").transition().duration(750).attr("cx", d => xScale(d.x))
        - Implement enter/exit animations: .enter().append("circle").attr("r", 0).transition().duration(500).attr("r", 5)

    11. Optimize performance:
        - Use d3.select() instead of d3.selectAll() when possible
        - Implement data joins efficiently: svg.selectAll(".dot").data(data, d => d.id)
        - Use requestAnimationFrame for smooth animations

    12. Ensure accessibility and readability:
        - Add ARIA labels: svg.attr("aria-label", "Data Visualization")
        - Use d3-textwrap for long labels: d3.textwrap().bounds([width: 100, height: 50])

    13. Implement zooming and panning:
        - Create zoom behavior: const zoom = d3.zoom().scaleExtent([1, 8]).on("zoom", zoomed)
        - Apply zoom: svg.call(zoom)

    14. Add crosshair effect for precise data reading:
        - Create crosshair lines: const crosshairX = svg.append("line"), crosshairY = svg.append("line")
        - Update on mouse move: svg.on("mousemove", updateCrosshair)

    15. Implement brush for range selection:
        - Create brush: const brush = d3.brushX().extent([[0, 0], [width, height]])
        - Apply brush: svg.append("g").attr("class", "brush").call(brush)

    16. Implement smooth color transitions:
        - Use d3.interpolateRgb for color transitions: .transition().duration(750).attr("fill", d => d3.interpolateRgb(oldColor, newColor)(t))

    17. Add engaging micro-interactions:
        - Implement click-to-lock tooltip: .on("click", lockTooltip)
        - Add hover effects on legend items: .on("mouseover", highlightCategory)

    18. Remember to comply with the user's request intelligently, updating existing code if it's an update request, or creating new code if it's a new visualization request. Always return the complete, updated code.

    19. The nature of visualization is comparative. So the user will be comparing multiple data sets. So the visualization must explicitly show the comparison and highlight the differences.

    20. You must understand how exactly the source data is different from each other and show the differences in the visualization intelligently by pointing out the differentiating factors.


    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    Previous D3 Code:
    {d3_code}

    User Request:
    {user_input}

    IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code. If the user request sounds like an update to the previous code, modify and return the entire updated code. Otherwise, generate and return entirely new code.
    """
    
    prompt = base_prompt
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a D3.js expert specializing in creating clear, readable, and comparative visualizations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        d3_code = response.choices[0].message.content
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

def display_visualization(d3_code: str) -> str:
    """
    Prepare the HTML content for the D3.js visualization.
    
    Args:
        d3_code (str): The D3.js code to be executed.
    
    Returns:
        str: HTML content for the visualization.
    """
    html_content = f"""
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://d3js.org/d3-scale-chromatic.v1.min.js"></script>
        <style>
            #visualization {{
                width: 100%;
                height: 100%;
                overflow: auto;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <script>
            const data = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
            const svgElement = d3.select("#visualization")
                .append("svg")
                .attr("width", "100%")
                .attr("height", "100%")
                .attr("viewBox", "0 0 1000 600")
                .attr("preserveAspectRatio", "xMidYMid meet")
                .node();
            
            {d3_code}
            
            createVisualization(data, svgElement);
        </script>
    </body>
    </html>
    """
    return html_content

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
            .attr("transform", `translate($[margin.left],$[margin.top])`);

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
            .attr("transform", `translate(0, $[height])`)
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
                    cleaned_d3_code = clean_d3_response(d3_code)
                st.session_state.current_viz = cleaned_d3_code
                st.session_state.workflow_history.append({
                    "version": len(st.session_state.workflow_history) + 1,
                    "request": user_query,
                    "code": cleaned_d3_code
                })
                st.rerun()
            else:
                st.warning("Please enter a visualization request.")

        if 'current_viz' in st.session_state:
            st.subheader("Current Visualization")
            try:
                with st.spinner("Preparing visualization..."):
                    html_content = display_visualization(st.session_state.current_viz)
                    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))
                    iframe_url = f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}"
                    st.components.v1.iframe(iframe_url, width=1000, height=600, scrolling=True)
            except Exception as e:
                st.error(f"An error occurred while displaying the visualization: {str(e)}")
                st.error("Please check the D3.js code for any issues.")
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

if __name__ == "__main__":
    main()
