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
from streamlit import components

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
if 'update_viz' not in st.session_state:
    st.session_state.update_viz = False  # Flag to trigger visualization update
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Stores the chat history

def display_loading_animation():
    loading_html = """
    <div class="loading-spinner" style="display: flex; justify-content: center; align-items: center; height: 500px;">
        <div class="spinner" style="border: 8px solid #f3f3f3; border-top: 8px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite;"></div>
    </div>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """
    return st.components.v1.html(loading_html, height=500)

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
    try:
        client = OpenAI(api_key=api_key)
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
        df1['source'] = 'CSV file 1'
        df2['source'] = 'CSV file 2'
        
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
    data_sample = df.head(5).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
    
    client = OpenAI(api_key=api_key)
    base_prompt = f"""
# D3.js Code Generation Task

Generate ONLY D3.js version 7 code for a sophisticated, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

Critical Requirements for D3.js Visualization:
1. Create a function named createVisualization(data, svgElement)
2. Implement a responsive SVG that adjusts its size based on the content
3. Utilize the full width and height provided (960x540 pixels)
4. Implement zoom and pan functionality for exploring the data
5. Ensure efficient use of space, minimizing empty areas
6. Add appropriate margins, title, axes labels, and a legend
7. Use different colors or patterns for each data source
8. Implement tooltips showing full information on hover
9. Include smooth transitions for any data updates or interactions
10. Handle potential overlapping of data points or labels
11. Always have grid lines on the visualization
12. Animate the visualization as much as possible
13. Utilize d3.select() for DOM manipulation and d3.data() for data binding
14. Implement advanced scales: d3.scaleLinear(), d3.scaleBand(), d3.scaleTime(), d3.scaleOrdinal(d3.schemeCategory10)
15. Create dynamic, animated axes using d3.axisBottom(), d3.axisLeft() with custom tick formatting
16. Implement smooth transitions and animations using d3.transition() and d3.easeCubic
17. Utilize d3.line(), d3.area(), d3.arc() for creating complex shapes and paths
18. Implement interactivity: d3.brush(), d3.zoom(), d3.drag() for user interaction
19. Use d3.interpolate() for smooth color and value transitions
20. Implement advanced layouts: d3.hierarchy(), d3.treemap(), d3.pack() for hierarchical data
21. Utilize d3.forceSimulation() for force-directed graph layouts
22. Implement d3.geoPath() and d3.geoProjection() for geographical visualizations
23. Use d3.contours() and d3.density2D() for density and contour visualizations
24. Implement d3.voronoi() for proximity-based visualizations
25. Utilize d3.chord() and d3.ribbon() for relationship visualizations
26. Implement advanced event handling with d3.on() for mouseover, click, etc.
27. Use d3.format() for number formatting in tooltips and labels
28. Implement d3.timeFormat() for date/time formatting
29. Utilize d3.range() and d3.shuffle() for data generation and randomization
30. Implement d3.nest() for data restructuring and aggregation
31. Use d3.queue() for asynchronous data loading and processing
32. Implement accessibility features using ARIA attributes and d3-textwrap
33. Optimize performance using d3.quadtree() for spatial indexing
34. Implement responsive design using d3.select(window).on("resize", ...)
35. Focus on creating a comparative visualization that highlights data differences
36. Implement error handling for invalid data formats and gracefully handle missing data
37. Create an interactive, filterable legend using d3.dispatch() for coordinated views
38. Implement crosshair functionality for precise data reading
39. Add a subtle, styled background using d3.select().append("rect") with rounded corners
40. Ensure the visualization updates smoothly when data changes or on user interaction
41. Use d3.transition().duration() to control animation speed, with longer durations for more complex animations
42. Implement staggered animations using d3.transition().delay() to create cascading effects
43. Utilize d3.easeElastic, d3.easeBack, or custom easing functions for more dynamic animations
44. Implement enter, update, and exit animations for data changes
45. Use d3.interpolateString() for smooth transitions between different text values
46. Implement path animations using d3.interpolate() for custom interpolators
47. Create looping animations using d3.timer() for continuous effects
48. Implement chained transitions using .transition().transition() for sequential animations
49. Use d3.active() to coordinate multiple animations and prevent overlapping
50. Implement FLIP (First, Last, Invert, Play) animations for layout changes

Data Schema:
{schema_str}

Sample Data:
{json.dumps(data_sample, indent=2)}

IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
"""

    if user_input:
        prompt = f"""
# D3.js Code Generation Task

Generate ONLY D3.js version 7 code for a clear, readable, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

Critical Requirements:
1. Create a function named createVisualization(data, svgElement)
2. Implement a visualization that explicitly compares data from two CSV files AND satisfies this user prompt:
---
{user_input}
---
3. Solve the overlapping labels problem:
   - Rotate labels if necessary (e.g., 45-degree angle)
   - Use a larger SVG size (e.g., width: 1000px, height: 600px) to accommodate all labels
   - Implement label truncation or abbreviation for long names
4. Use different colors or patterns for each data source
5. Include a legend clearly indicating which color/pattern represents which data source
6. Ensure appropriate spacing between bars or data points
7. Add tooltips showing full information on hover
8. Implement responsive design to fit various screen sizes
9. Include smooth transitions for any data updates

Data Schema:
{schema_str}

Sample Data:
{json.dumps(data_sample, indent=2)}

Current Code:```javascript
{st.session_state.current_viz}
IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code. """ else: prompt = base_prompt

try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a D3.js expert specializing in creating clear, readable, and comparative visualizations. Your code must explicitly address overlapping labels and ensure a comparative aspect between two data sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000
    )
    
    d3_code = response.choices[0].message.content.strip()
    if not d3_code:
        raise ValueError("Generated D3 code is empty")
    
    return d3_code
except Exception as e:
    logger.error(f"Error generating D3 code: {str(e)}")
    return generate_fallback_visualization()
def refine_d3_code(initial_code: str, api_key: str, max_attempts: int = 3) -> str: """ Refine the D3 code through iterative LLM calls if necessary.


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

Defines a createVisualization(data, svgElement) function
Uses only D3.js version 7 syntax
Creates a valid visualization
Return ONLY the corrected D3 code without any explanations or comments. """ try: response = client.chat.completions.create( model="gpt-4", messages=[ {"role": "system", "content": "You are a D3.js expert. Provide only valid D3 code."}, {"role": "user", "content": refinement_prompt} ], temperature=0.3, max_tokens=1500 )


        initial_code = clean_d3_response(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error refining D3 code: {str(e)}")
        break

# If we've exhausted our attempts, return the last attempt
logger.warning("Failed to generate valid D3 code after maximum attempts")
return initial_code
def clean_d3_response(response: str) -> str: """ Clean the LLM response to ensure it only contains D3 code.


This function removes markdown formatting, non-JavaScript lines,
and ensures the code starts with the createVisualization function.

Args:
    response (str): The raw response from the LLM.

Returns:
    str: Cleaned D3.js code.
"""
# Remove any potential markdown code blocks
response = response.replace("```javascript", "").replace("```", "").strip()

# Remove any lines that don't look like JavaScript
clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]

# Ensure the code starts with the createVisualization function
if not any(line.strip().startswith('function createVisualization') for line in clean_lines):
    clean_lines.insert(0, 'function createVisualization(data, svgElement) {')
    clean_lines.append('}')

return '\n'.join(clean_lines)

def get_html_template():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>D3 Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; }
        #visualization { width: 100%; height: 100vh; overflow: hidden; }
        svg { width: 100%; height: 100%; }
        .tooltip {
            position: absolute;
            text-align: left;
            width: auto;
            padding: 10px;
            font: 12px sans-serif;
            background: rgba(0,0,0,0.7);
            color: #fff;
            border: 0px;
            border-radius: 8px;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div id="visualization"></div>
    <button onclick="downloadSVG()" style="position: absolute; top: 10px; right: 10px; padding: 10px 20px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">Download SVG</button>
    <script>
        {d3_code}
        // Create the SVG element
        const svgElement = d3.select("#visualization")
            .append("svg")
            .attr("viewBox", "0 0 960 540")
            .attr("preserveAspectRatio", "xMidYMid meet")
            .node();

        // Get the data from the parent window
        const vizData = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
        
        // Call the createVisualization function
        createVisualization(vizData, svgElement);

        // Function to download the SVG
        function downloadSVG() {{
            const svgData = new XMLSerializer().serializeToString(svgElement);
            const svgBlob = new Blob([svgData], {{type: "image/svg+xml;charset=utf-8"}});
            const svgUrl = URL.createObjectURL(svgBlob);
            const downloadLink = document.createElement("a");
            downloadLink.href = svgUrl;
            downloadLink.download = "visualization.svg";
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        }}

        // Make the visualization responsive
        window.addEventListener('resize', function() {{
            d3.select(svgElement)
                .attr("viewBox", "0 0 " + window.innerWidth + " " + window.innerHeight);
        }});
    </script>
</body>
</html>
"""

def display_visualization(d3_code: str):
    """ Display the D3.js visualization using an iframe and add a download button. """
    html_content = get_html_template().format(d3_code=d3_code)

    # Encode the data to pass it to the iframe
    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))

    # Display the iframe with the encoded data in the URL hash
    st.components.v1.iframe(
        f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}", 
        width=960, 
        height=600, 
        scrolling=False
    )

def generate_fallback_visualization() -> str: """ Generate a fallback visualization if the LLM fails.


This function creates a simple scatter plot using D3.js as a fallback
when the main visualization generation process fails.

Returns:
    str: D3.js code for a simple scatter plot visualization.
"""
logger.info("Generating fallback visualization")

fallback_code = """
function createVisualization(data, svgElement) {
    const margin = { top: 50, right: 30, bottom: 60, left: 70 },
        width = 960 - margin.left - margin.right,
        height = 540 - margin.top - margin.bottom;
    
    const svg = d3.select(svgElement)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
    
    const xScale = d3.scaleLinear()
        .domain([d3.min(data, d => +d.sepal_width), d3.max(data, d => +d.sepal_width)])
        .range([0, width]);
    
    const yScale = d3.scaleLinear()
        .domain([d3.min(data, d => +d.sepal_length), d3.max(data, d => +d.sepal_length)])
        .range([height, 0]);
    
    const colorScale = d3.scaleOrdinal()
        .domain([...new Set(data.map(d => d.source))])
        .range(["#1f77b4", "#ff7f0e"]);
    
    // Add X axis
    svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale));
    
    // Add Y axis
    svg.append("g")
        .call(d3.axisLeft(yScale));
    
    // Add gridlines
    svg.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale)
            .tickSize(-height)
            .tickFormat(""));
    
    svg.append("g")
        .attr("class", "grid")
        .call(d3.axisLeft(yScale)
            .tickSize(-width)
            .tickFormat(""));
    
    // Create tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);
    
    // Draw dots
    svg.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", d => xScale(+d.sepal_width))
        .attr("cy", d => yScale(+d.sepal_length))
        .attr("r", 5)
        .style("fill", d => colorScale(d.source))
        .style("opacity", 0.7)
        .on("mouseover", function(event, d) {
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`Species: ${d.species}<br/>Source: ${d.source}<br/>Sepal Length: ${d.sepal_length}<br/>Sepal Width: ${d.sepal_width}`)
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function() {
            tooltip.transition().duration(500).style("opacity", 0);
        });
    
    // Add legend
    const legend = svg.selectAll(".legend")
        .data(colorScale.domain())
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", (d, i) => `translate(0,${i * 20})`);
    
    legend.append("rect")
        .attr("x", width - 18)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", colorScale);
    
    legend.append("text")
        .attr("x", width - 24)
        .attr("y", 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(d => d);
}
"""

logger.info("Fallback visualization generated successfully")
return fallback_code
def generate_and_validate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str: """Generate, validate, and if necessary, refine D3 code.""" initial_code = generate_d3_code(df, api_key, user_input) cleaned_code = clean_d3_response(initial_code)


if validate_d3_code(cleaned_code):
    return cleaned_code
else:
    return refine_d3_code(cleaned_code, api_key)
def main(): st.set_page_config(page_title="🎨 Comparative Visualization Generator", page_icon="✨", layout="wide") st.title("🎨 Comparative Visualization Generator")


api_key = get_api_key()

if api_key and not test_api_key(api_key):
    st.error("Invalid OpenAI API Key. Please check and try again.")
    return

st.header("Upload CSV Files")
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload first CSV file", type="csv")
with col2:
    file2 = st.file_uploader("Upload second CSV file", type="csv")

if file1 and file2:
    try:
        if 'preprocessed_df' not in st.session_state or st.session_state.preprocessed_df is None:
            with st.spinner("Preprocessing data..."):
                merged_df = preprocess_data(file1, file2)
            st.session_state.preprocessed_df = merged_df
        
        with st.expander("Preview of Preprocessed Data"):
            st.dataframe(st.session_state.preprocessed_df.head())
        
        if 'current_viz' not in st.session_state or st.session_state.current_viz is None:
            with st.spinner("Generating initial D3 visualization..."):
                d3_code = generate_and_validate_d3_code(st.session_state.preprocessed_df, api_key)
                st.session_state.current_viz = d3_code
                st.session_state.workflow_history.append({
                    "version": len(st.session_state.workflow_history) + 1,
                    "request": "Initial comparative visualization",
                    "code": d3_code
                })
        
        # Create a placeholder for the visualization
        viz_placeholder = st.empty()
        
        # Display the current visualization
        with viz_placeholder.container():
            st.subheader("Current Visualization")
            display_visualization(st.session_state.current_viz)
        
        st.subheader("Modify Visualization")
        user_input = st.text_area("Enter your modification request (or type 'exit' to finish):", height=100)
        
        if st.button("Update Visualization"):
            if user_input.lower().strip() == 'exit':
                st.success("Visualization process completed.")
            elif user_input:
                # Replace current visualization with loading animation
                with viz_placeholder.container():
                    st.subheader("Updating Visualization")
                    display_loading_animation()
                
                # Generate new visualization
                with st.spinner("Generating updated visualization..."):
                    modified_d3_code = generate_and_validate_d3_code(st.session_state.preprocessed_df, api_key, user_input)
                    st.session_state.current_viz = modified_d3_code
                    st.session_state.workflow_history.append({
                        "version": len(st.session_state.workflow_history) + 1,
                        "request": user_input,
                        "code": modified_d3_code
                    })
                
                # Update the visualization in place
                with viz_placeholder.container():
                    st.subheader("Current Visualization")
                    display_visualization(st.session_state.current_viz)
            else:
                st.warning("Please enter a modification request or type 'exit' to finish.")
        
        with st.expander("View/Edit Visualization Code"):
            code_editor = st.text_area("D3.js Code", value=st.session_state.current_viz, height=300, key="code_editor")
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                edit_enabled = st.checkbox("Enable Edit", key="edit_toggle")
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
                            # Update the visualization in place
                            with viz_placeholder.container():
                                st.subheader("Current Visualization")
                                display_visualization(st.session_state.current_viz)
                        else:
                            st.error("Invalid D3.js code. Please check your code and try again.")
                    else:
                        st.warning("Enable 'Edit' to make changes.")
            with col3:
                if st.button("Copy Code"):
                    st.write("Code copied to clipboard!")
                    # Using JavaScript to copy text to clipboard
                    st.write(f'''
                        <textarea id="code-to-copy" style="position: absolute; left: -9999px;">{code_editor}</textarea>
                        <script>
                            const textArea = document.getElementById('code-to-copy');
                            textArea.select();
                            document.execCommand('copy');
                        </script>
                    ''', unsafe_allow_html=True)
        
        with st.expander("Workflow History"):
            for i, step in enumerate(st.session_state.workflow_history):
                with st.expander(f"Step {i+1}: {step['request']}"):
                    st.code(step['code'], language='javascript')
                    if st.button(f"Revert to Step {i+1}"):
                        st.session_state.current_viz = step['code']
                        # Update the visualization in place
                        with viz_placeholder.container():
                            st.subheader("Current Visualization")
                            display_visualization(st.session_state.current_viz)
                        st.success(f"Reverted to Step {i+1}.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("An unexpected error occurred. Please try again or contact support if the problem persists.")
        st.code(traceback.format_exc())  # Display traceback for debugging
else:
    st.info("Please upload both CSV files to visualize your data")
if name == "main": main()
