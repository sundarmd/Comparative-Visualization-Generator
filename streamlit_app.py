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

def clean_d3_response(response: str) -> str:
    # Remove any code block markers
    response = re.sub(r'```[a-z]*', '', response)
    response = response.replace('```', '')
    # Remove any leading/trailing quotes
    response = response.strip('\'" \n')
    return response

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
    15. Focus on creating a comparative visualization that highlights data differences

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
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a D3.js expert specializing in creating clear, readable, and comparative visualizations. Your code must explicitly address overlapping labels and ensure a comparative aspect between two data sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        d3_code = response.choices[0].message.content
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")

        d3_code = clean_d3_response(d3_code)

        return d3_code
    except Exception as e:
        logger.error(f"Error generating D3 code: {str(e)}")
        st.error(f"Error generating D3 code:\n\n{str(e)}")
        raise

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
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a D3.js expert. Provide only valid D3 code."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0
            )

            initial_code = response.choices[0].message.content
            initial_code = clean_d3_response(initial_code)
        except Exception as e:
            logger.error(f"Error refining D3 code: {str(e)}")
            st.error(f"Error refining D3 code:\n\n{str(e)}")
            raise

    logger.warning("Failed to generate valid D3 code after maximum attempts")
    return initial_code

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

def generate_and_validate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    try:
        d3_code = generate_d3_code(df, api_key, user_input)
        if not validate_d3_code(d3_code):
            d3_code = refine_d3_code(d3_code, api_key)
        return d3_code
    except Exception as e:
        logger.error(f"Error in generate_and_validate_d3_code: {str(e)}")
        st.error(f"Error in generate_and_validate_d3_code:\n\n{str(e)}")  # Display error to user
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
                    st.error(f"Error generating initial visualization:\n\n{str(e)}")
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
                        st.error(f"Error updating visualization:\n\n{str(e)}")
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
            st.error(f"An error occurred:\n\n{str(e)}")
            logger.error(f"Error in main function: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("An unexpected error occurred. Please try again or contact support if the problem persists.")
            st.code(traceback.format_exc())
    else:
        st.info("Please upload both CSV files to visualize your data")

if __name__ == "__main__":
    main()
