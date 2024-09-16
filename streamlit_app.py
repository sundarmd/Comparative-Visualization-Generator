import streamlit as st
import pandas as pd
import openai
import os
import json
import logging
import traceback
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define MAX_WORKFLOW_HISTORY constant
MAX_WORKFLOW_HISTORY = 20

def display_loading_animation(message="Loading..."):
    loading_html = f"""
    <div class="loading-spinner" style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">
        <div class="spinner" style="border: 8px solid #f3f3f3; border-top: 8px solid #3498db; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite;"></div>
        <p style="margin-top: 20px; font-size: 1.2em;">{message}</p>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .loading-spinner {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
    </style>
    """
    st.markdown(loading_html, unsafe_allow_html=True)

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

def get_api_key() -> str:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.sidebar.warning("It's recommended to use environment variables or Streamlit secrets for API keys.")
    return api_key

def preprocess_data(file1, file2) -> pd.DataFrame:
    logger.info("Starting data preprocessing")
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'
        merged_df = pd.concat([df1, df2], ignore_index=True)
        merged_df = merged_df.fillna(0)
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                try:
                    merged_df[col] = pd.to_numeric(merged_df[col])
                except ValueError:
                    pass
        merged_df.columns = merged_df.columns.str.lower().str.replace(' ', '_')
        logger.info("Data preprocessing completed successfully")
        return merged_df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def validate_d3_code(code: str) -> bool:
    if not re.search(r'function\s+createVisualization\s*\(data,\s*svgElement\)\s*{', code):
        return False
    d3_methods = ['d3.select', 'd3.scaleLinear', 'd3.axisBottom', 'd3.axisLeft']
    if not any(method in code for method in d3_methods):
        return False
    if code.count('{') != code.count('}'):
        return False
    return True

def generate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    logger.info("Starting D3 code generation")
    data_sample = df.head(50).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
    openai.api_key = api_key

    base_prompt = f"""
    # D3.js Code Generation Task

    Generate ONLY D3.js version 7 code for a sophisticated, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

    Critical Requirements for D3.js Visualization:
    1. Create a function named createVisualization(data, svgElement)
    2. Always add an appropriate title to the visualization
    3. Always have grid lines on the visualization
    4. Always have a legend on the visualization that displays the color/pattern for each data source and category
    5. Always have tooltips on the visualization that display the full information on hover
    6. Always animate the visualization as much as possible
    7. Implement a responsive SVG canvas with margins: const svgWidth = 1200, svgHeight = 700
    8. Utilize d3.select() for DOM manipulation and d3.data() for data binding
    9. Implement advanced scales: d3.scaleLinear(), d3.scaleBand(), d3.scaleTime(), d3.scaleOrdinal(d3.schemeCategory10)
    10. Create dynamic, animated axes using d3.axisBottom(), d3.axisLeft() with custom tick formatting
    11. Implement smooth transitions and animations using d3.transition() and d3.easeCubic
    12. Utilize d3.line(), d3.area(), d3.arc() for creating complex shapes and paths
    13. Implement interactivity: d3.brush(), d3.zoom(), d3.drag() for user interaction
    14. Use d3.interpolate() for smooth color and value transitions
    15. Implement advanced layouts: d3.hierarchy(), d3.treemap(), d3.pack() for hierarchical data
    16. Utilize d3.forceSimulation() for force-directed graph layouts
    17. Implement d3.geoPath() and d3.geoProjection() for geographical visualizations
    18. Use d3.contours() and d3.density2D() for density and contour visualizations
    19. Implement d3.voronoi() for proximity-based visualizations
    20. Utilize d3.chord() and d3.ribbon() for relationship visualizations
    21. Implement advanced event handling with d3.on() for mouseover, click, etc.
    22. Use d3.format() for number formatting in tooltips and labels
    23. Implement d3.timeFormat() for date/time formatting
    24. Utilize d3.range() and d3.shuffle() for data generation and randomization
    25. Implement d3.nest() for data restructuring and aggregation
    26. Use d3.queue() for asynchronous data loading and processing
    27. Implement accessibility features using ARIA attributes and d3-textwrap
    28. Optimize performance using d3.quadtree() for spatial indexing
    29. Implement responsive design using d3.select(window).on("resize", ...)
    30. Focus on creating a comparative visualization that highlights data differences
    31. Implement error handling for invalid data formats and gracefully handle missing data
    32. Create an interactive, filterable legend using d3.dispatch() for coordinated views
    33. Implement crosshair functionality for precise data reading
    34. Add a subtle, styled background using d3.select().append("rect") with rounded corners
    35. Ensure the visualization updates smoothly when data changes or on user interaction
    36. Use d3.transition().duration() to control animation speed, with longer durations for more complex animations
    37. Implement staggered animations using d3.transition().delay() to create cascading effects
    38. Utilize d3.easeElastic, d3.easeBack, or custom easing functions for more dynamic animations
    39. Implement enter, update, and exit animations for data changes
    40. Use d3.interpolateString() for smooth transitions between different text values
    41. Implement path animations using d3.interpolate() for custom interpolators
    42. Create looping animations using d3.timer() for continuous effects
    43. Implement chained transitions using .transition().transition() for sequential animations
    44. Use d3.active() to coordinate multiple animations and prevent overlapping
    45. Implement FLIP (First, Last, Invert, Play) animations for layout changes

    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
    """

    if user_input:
        prompt = f"""
        # D3.js Code Generation Task

        Generate ONLY D3.js version 7 code for a sophisticated, interactive, and comparative visualization. Do not include any explanations, comments, or markdown formatting.

        Critical Requirements for D3.js Visualization:
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
        {json.dumps(data_sample[:5], indent=2)}

        Current Code:
        ```javascript
        {st.session_state.current_viz}
        ```

        IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
        """
    else:
        prompt = base_prompt

    try:
        response = openai.ChatCompletion.create(
            model="GPT-4o",  # Changed from 'gpt-4' to 'gpt-3.5-turbo'
            messages=[
                {"role": "system", "content": "You are a D3.js expert specializing in creating clear, readable, and comparative visualizations. Your code must explicitly address overlapping labels and ensure a comparative aspect between two data sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        d3_code = response['choices'][0]['message']['content']
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")

        return d3_code
    except Exception as e:
        logger.error(f"Error generating D3 code: {str(e)}")
        st.error(f"Error generating D3 code: {str(e)}")  # Display error to user
        raise  # Re-raise the exception to be caught in the main function

def refine_d3_code(initial_code: str, api_key: str, max_attempts: int = 3) -> str:
    openai.api_key = api_key

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

        try:
            response = openai.ChatCompletion.create(
                model="GPT-4o",  # Changed from 'gpt-4' to 'gpt-3.5-turbo'
                messages=[
                    {"role": "system", "content": "You are a D3.js expert. Provide only valid D3 code."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0
            )

            initial_code = clean_d3_response(response['choices'][0]['message']['content'])
        except Exception as e:
            logger.error(f"Error refining D3 code: {str(e)}")
            st.error(f"Error refining D3 code: {str(e)}")  # Display error to user
            raise  # Re-raise the exception to be caught in the main function

    logger.warning("Failed to generate valid D3 code after maximum attempts")
    return initial_code

def clean_d3_response(response: str) -> str:
    response = response.replace("```javascript", "").replace("```", "")
    clean_lines = [line for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
    if not any(line.strip().startswith('function createVisualization') for line in clean_lines):
        clean_lines.insert(0, 'function createVisualization(data, svgElement) {')
        clean_lines.append('}')
    return '\n'.join(clean_lines)

def display_visualization(d3_code: str):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
            #visualization {{
                width: 100%;
                height: 100vh;
            }}
            svg {{
                width: 100%;
                height: 100%;
            }}
            button {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 10;
                padding: 10px 20px;
                font-size: 16px;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <button onclick="downloadSVG()">Download SVG</button>
        <script>
            {d3_code}
            const svgElement = d3.select("#visualization")
                .append("svg")
                .attr("viewBox", "0 0 1200 700")
                .attr("preserveAspectRatio", "xMidYMid meet")
                .node();
            const vizData = {json.dumps(st.session_state.preprocessed_df.to_dict(orient='records'))};
            createVisualization(vizData, d3.select(svgElement));
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
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_content, height=800, scrolling=True)

def generate_fallback_visualization() -> str:
    logger.info("Generating fallback visualization")
    fallback_code = """
    function createVisualization(data, svgElement) {
        const margin = { top: 20, right: 20, bottom: 50, left: 50 };
        const width = 1200 - margin.left - margin.right;
        const height = 700 - margin.top - margin.bottom;

        const svg = svgElement
            .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

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
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end");

        svg.append("g")
            .call(d3.axisLeft(yScale));

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", -10)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text("Fallback Visualization");

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + margin.bottom - 5)
            .attr("text-anchor", "middle")
            .text(xKey);

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -margin.left + 15)
            .attr("text-anchor", "middle")
            .text(yKey);
    }
    """
    logger.info("Fallback visualization generated successfully")
    return fallback_code

def generate_and_validate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    try:
        d3_code = generate_d3_code(df, api_key, user_input)
        if not validate_d3_code(d3_code):
            d3_code = refine_d3_code(d3_code, api_key)
        return d3_code
    except Exception as e:
        logger.error(f"Error in generate_and_validate_d3_code: {str(e)}")
        st.error(f"Error in generate_and_validate_d3_code: {str(e)}")  # Display error to user
        raise  # Re-raise the exception to be caught in the main function

def main():
    st.set_page_config(page_title="🎨 Comparative Visualization Generator", page_icon="✨", layout="wide")
    st.title("🎨 Comparative Visualization Generator")

    api_key = get_api_key()

    st.header("Upload CSV Files")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload first CSV file", type="csv", key="file1")
    with col2:
        file2 = st.file_uploader("Upload second CSV file", type="csv", key="file2")

    if 'update_viz' not in st.session_state:
        st.session_state.update_viz = False

    if file1 and file2:
        try:
            if 'preprocessed_df' not in st.session_state or st.session_state.preprocessed_df is None:
                with st.spinner("Preprocessing data..."):
                    merged_df = preprocess_data(file1, file2)
                st.session_state.preprocessed_df = merged_df

            with st.expander("Preview of preprocessed data"):
                st.dataframe(st.session_state.preprocessed_df.head())

            viz_placeholder = st.empty()

            if 'current_viz' not in st.session_state or st.session_state.current_viz is None:
                with viz_placeholder.container():
                    display_loading_animation("Generating initial visualization...")
                try:
                    initial_d3_code = generate_and_validate_d3_code(st.session_state.preprocessed_df, api_key)
                    st.session_state.current_viz = initial_d3_code
                    st.session_state.workflow_history = [{
                        "version": 1,
                        "request": "Initial visualization",
                        "code": initial_d3_code
                    }]
                except Exception as e:
                    st.error(f"Error generating initial visualization: {str(e)}")
                    # Optionally, you can decide whether to display the fallback visualization or not
                    # st.session_state.current_viz = generate_fallback_visualization()
                finally:
                    if st.session_state.current_viz:
                        with viz_placeholder.container():
                            st.subheader("Current Visualization")
                            display_visualization(st.session_state.current_viz)
            else:
                with viz_placeholder.container():
                    st.subheader("Current Visualization")
                    display_visualization(st.session_state.current_viz)

            st.subheader("Modify Visualization")
            user_input = st.text_area("Enter your modification request (or type 'exit' to finish):", height=100)

            if st.button("Update Visualization"):
                if user_input.lower().strip() == 'exit':
                    st.success("Visualization process completed.")
                elif user_input:
                    with viz_placeholder.container():
                        display_loading_animation("Generating updated visualization...")
                    try:
                        modified_d3_code = generate_and_validate_d3_code(st.session_state.preprocessed_df, api_key, user_input)
                        st.session_state.current_viz = modified_d3_code
                        st.session_state.workflow_history.append({
                            "version": len(st.session_state.workflow_history) + 1,
                            "request": user_input,
                            "code": modified_d3_code
                        })
                        if len(st.session_state.workflow_history) > MAX_WORKFLOW_HISTORY:
                            st.session_state.workflow_history.pop(0)
                    except Exception as e:
                        st.error(f"Error updating visualization: {str(e)}")
                    finally:
                        if st.session_state.current_viz:
                            with viz_placeholder.container():
                                st.subheader("Current Visualization")
                                display_visualization(st.session_state.current_viz)
                else:
                    st.warning("Please enter a modification request or type 'exit' to finish.")

            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area("D3.js Code", value=st.session_state.current_viz or "", height=300, key="code_editor")
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    edit_enabled = st.checkbox("Edit", key="edit_toggle")
                with col2:
                    if st.button("Execute Code"):
                        if edit_enabled:
                            if validate_d3_code(code_editor):
                                st.session_state.current_viz = code_editor
                                st.session_state.workflow_history.append({
                                    "request": "Manual code edit",
                                    "code": code_editor
                                })
                                if len(st.session_state.workflow_history) > MAX_WORKFLOW_HISTORY:
                                    st.session_state.workflow_history.pop(0)
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
                        st.write(f'<textarea style="position: absolute; left: -9999px;">{code_editor}</textarea>', unsafe_allow_html=True)
                        st.write('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)

            with st.expander("Workflow History"):
                for i, step in enumerate(st.session_state.workflow_history):
                    st.subheader(f"Step {i+1}")
                    st.write(f"Request: {step['request']}")
                    if st.button(f"Revert to Step {i+1}", key=f"revert_{i}"):
                        st.session_state.current_viz = step['code']
                        with viz_placeholder.container():
                            st.subheader("Current Visualization")
                            display_visualization(st.session_state.current_viz)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main function: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("An unexpected error occurred. Please try again or contact support if the problem persists.")
            st.code(traceback.format_exc())
    else:
        st.info("Please upload both CSV files to visualize your data")

if __name__ == "__main__":
    main()
