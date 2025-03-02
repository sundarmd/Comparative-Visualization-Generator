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
from streamlit import components
from dotenv import load_dotenv

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
if 'update_viz' not in st.session_state:
    st.session_state.update_viz = False  # Flag to trigger visualization update
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Stores the chat history

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
    Test if the provided API key is valid.
    
    This function attempts to make a simple API call using the provided OpenAI API key.
    If successful, the key is considered valid.
    
    Args:
        api_key (str): The OpenAI API key to test.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    try:
        openai.api_key = api_key
        model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")
        
        openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
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
    
    # Return dictionary with validation results
    return {
        "valid": len(missing_features) == 0,
        "missing_features": missing_features
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

def get_visualization_template(viz_type: str) -> str:
    """
    Get a template for a specific visualization type.
    
    Args:
        viz_type (str): The type of visualization (scatterplot, histogram, parallel, etc.)
    
    Returns:
        str: Template code for the specified visualization type.
    """
    templates = {
        "scatterplot": """
function createVisualization(data, svgElement) {
  // Configuration
  const config = {
    margin: { top: 60, right: 120, bottom: 80, left: 80 },
    width: 960,
    height: 600,
    transitionDuration: 800,
    colors: d3.scaleOrdinal(d3.schemeCategory10),
    tooltipDelay: 300,
    pointRadius: 5,
    pointPadding: 1.5,
    gridOpacity: 0.15,
    brushHeight: 60,
    zoomExtent: [0.5, 10],
    animationEasing: d3.easeCubicInOut
  };

  // Responsive dimensions
  const containerWidth = parseInt(d3.select(svgElement.node().parentNode).style('width'));
  const containerHeight = parseInt(d3.select(svgElement.node().parentNode).style('height'));
  const width = (containerWidth || config.width) - config.margin.left - config.margin.right;
  const height = (containerHeight || config.height) - config.margin.top - config.margin.bottom;

  // Clear previous visualization
  svgElement.selectAll("*").remove();

  // Setup SVG with proper dimensions and viewBox for responsiveness
  svgElement
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", `0 0 ${width + config.margin.left + config.margin.right} ${height + config.margin.top + config.margin.bottom}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  // Create main visualization group with margin
  const svg = svgElement.append("g")
    .attr("transform", `translate(${config.margin.left},${config.margin.top})`)
    .attr("class", "main-viz-group");

  // Add styled background
  svg.append("rect")
    .attr("width", width)
    .attr("height", height)
    .attr("fill", "#f9f9f9")
    .attr("rx", 8)
    .attr("ry", 8)
    .attr("filter", "drop-shadow(0px 2px 3px rgba(0,0,0,0.1))");

  // Data analysis and preparation
  const sourceGroups = d3.group(data, d => d.source || d.species);
  const sources = Array.from(sourceGroups.keys());
  
  // Detect numeric columns for dropdown options
  const numericColumns = Object.entries(data[0])
    .filter(([key, value]) => !isNaN(+value) && key !== 'source' && key !== 'species')
    .map(([key]) => key);
  
  // Default axes (allow user to change via UI)
  let xKey = numericColumns[0] || Object.keys(data[0])[0];
  let yKey = numericColumns[1] || Object.keys(data[0])[1];
  let sizeKey = numericColumns[2] || numericColumns[0];
  
  // Create scales with domains based on data
  const xScale = d3.scaleLinear()
    .domain([d3.min(data, d => +d[xKey]) * 0.9, d3.max(data, d => +d[xKey]) * 1.1])
    .range([0, width]);
  
  const yScale = d3.scaleLinear()
    .domain([d3.min(data, d => +d[yKey]) * 0.9, d3.max(data, d => +d[yKey]) * 1.1])
    .range([height, 0]);
  
  // Size scale for data points
  const sizeScale = d3.scaleLinear()
    .domain([d3.min(data, d => +d[sizeKey]), d3.max(data, d => +d[sizeKey])])
    .range([3, 12]);
  
  // Create axes with grid lines
  const xAxis = d3.axisBottom(xScale)
    .tickSize(-height)
    .tickPadding(10)
    .ticks(10)
    .tickFormat(d3.format(".2f"));
  
  const yAxis = d3.axisLeft(yScale)
    .tickSize(-width)
    .tickFormat(d3.format(".2f"))
    .tickPadding(10);
  
  // Add X axis with animation
  const xAxisGroup = svg.append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${height})`)
    .call(xAxis);
  
  // Style X axis
  xAxisGroup.selectAll(".tick line")
    .attr("stroke", "#ddd")
    .attr("opacity", config.gridOpacity);
  
  xAxisGroup.selectAll(".tick text")
    .attr("font-size", "12px")
    .attr("font-family", "Arial");
  
  xAxisGroup.select(".domain")
    .attr("stroke", "#999");
  
  // Add Y axis with animation
  const yAxisGroup = svg.append("g")
    .attr("class", "y-axis")
    .call(yAxis);
  
  // Style Y axis
  yAxisGroup.selectAll(".tick line")
    .attr("stroke", "#ddd")
    .attr("opacity", config.gridOpacity);
  
  yAxisGroup.selectAll(".tick text")
    .attr("font-size", "12px")
    .attr("font-family", "Arial");
  
  yAxisGroup.select(".domain")
    .attr("stroke", "#999");
  
  // Add axis labels
  const xLabel = svg.append("text")
    .attr("class", "x-axis-label")
    .attr("x", width / 2)
    .attr("y", height + 60)
    .attr("text-anchor", "middle")
    .attr("font-size", "14px")
    .attr("font-weight", "bold")
    .attr("fill", "#555")
    .text(xKey);
  
  const yLabel = svg.append("text")
    .attr("class", "y-axis-label")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("y", -60)
    .attr("text-anchor", "middle")
    .attr("font-size", "14px")
    .attr("font-weight", "bold")
    .attr("fill", "#555")
    .text(yKey);
  
  // Add title with animation
  const title = svg.append("text")
    .attr("class", "chart-title")
    .attr("x", width / 2)
    .attr("y", -30)
    .attr("text-anchor", "middle")
    .attr("font-size", "20px")
    .attr("font-weight", "bold")
    .attr("fill", "#333")
    .text(`Scatterplot of ${yKey} vs ${xKey}`)
    .style("opacity", 0)
    .transition()
    .duration(1000)
    .style("opacity", 1);
  
  // Create tooltip
  const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("position", "absolute")
    .style("background", "rgba(255, 255, 255, 0.95)")
    .style("padding", "10px")
    .style("border-radius", "5px")
    .style("box-shadow", "0 0 10px rgba(0,0,0,0.25)")
    .style("pointer-events", "none")
    .style("font-family", "Arial")
    .style("font-size", "12px")
    .style("z-index", "10")
    .style("opacity", 0);
  
  // Create a clip path for the chart area
  svg.append("defs").append("clipPath")
    .attr("id", "clip")
    .append("rect")
    .attr("width", width)
    .attr("height", height);
  
  // Create chart area with clip path
  const chartArea = svg.append("g")
    .attr("clip-path", "url(#clip)")
    .attr("class", "chart-area");
  
  // Create points with animations and interactions
  const points = chartArea.selectAll(".point")
    .data(data)
    .enter()
    .append("circle")
    .attr("class", "point")
    .attr("cx", d => xScale(+d[xKey]))
    .attr("cy", height) // Start from bottom
    .attr("r", d => sizeScale(+d[sizeKey]))
    .attr("fill", d => config.colors(d.source || d.species))
    .attr("stroke", "#fff")
    .attr("stroke-width", 1)
    .style("cursor", "pointer")
    .on("mouseover", function(event, d) {
      d3.select(this)
        .transition()
        .duration(300)
        .attr("fill", d3.color(config.colors(d.source || d.species)).brighter(0.5))
        .attr("stroke-width", 2)
        .attr("r", d => sizeScale(+d[sizeKey]) * 1.5);
      
      tooltip.transition()
        .duration(200)
        .style("opacity", 0.9);
      
      // Format all data properties for tooltip
      const tooltipContent = Object.entries(d)
        .map(([key, value]) => `<strong>${key}:</strong> ${value}`)
        .join("<br>");
      
      tooltip.html(tooltipContent)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
      
      // Add crosshair
      chartArea.append("line")
        .attr("class", "crosshair-x")
        .attr("x1", xScale(+d[xKey]))
        .attr("x2", xScale(+d[xKey]))
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "#999")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "5,5");
      
      chartArea.append("line")
        .attr("class", "crosshair-y")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", yScale(+d[yKey]))
        .attr("y2", yScale(+d[yKey]))
        .attr("stroke", "#999")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "5,5");
    })
    .on("mouseout", function(event, d) {
      d3.select(this)
        .transition()
        .duration(300)
        .attr("fill", config.colors(d.source || d.species))
        .attr("stroke-width", 1)
        .attr("r", d => sizeScale(+d[sizeKey]));
      
      tooltip.transition()
        .duration(500)
        .style("opacity", 0);
      
      // Remove crosshair
      chartArea.selectAll(".crosshair-x, .crosshair-y").remove();
    })
    .on("click", function(event, d) {
      // Show detailed information
      const detailsDiv = d3.select("body").append("div")
        .attr("class", "details-popup")
        .style("position", "fixed")
        .style("left", "50%")
        .style("top", "50%")
        .style("transform", "translate(-50%, -50%)")
        .style("background", "white")
        .style("padding", "20px")
        .style("border-radius", "10px")
        .style("box-shadow", "0 0 20px rgba(0,0,0,0.3)")
        .style("z-index", "1000")
        .style("max-width", "500px")
        .style("width", "80%");
      
      detailsDiv.append("h3")
        .text(`Details for Point`);
      
      const table = detailsDiv.append("table")
        .style("width", "100%")
        .style("border-collapse", "collapse");
      
      Object.entries(d).forEach(([key, value]) => {
        const row = table.append("tr");
        row.append("td")
          .text(key)
          .style("padding", "8px")
          .style("border-bottom", "1px solid #ddd")
          .style("font-weight", "bold");
        
        row.append("td")
          .text(value)
          .style("padding", "8px")
          .style("border-bottom", "1px solid #ddd");
      });
      
      detailsDiv.append("button")
        .text("Close")
        .style("margin-top", "15px")
        .style("padding", "8px 15px")
        .style("background", "#f44336")
        .style("color", "white")
        .style("border", "none")
        .style("border-radius", "4px")
        .style("cursor", "pointer")
        .on("click", function() {
          detailsDiv.remove();
        });
    })
    .transition()
    .duration(config.transitionDuration)
    .delay((d, i) => i * 10) // Staggered animation
    .attr("cy", d => yScale(+d[yKey]))
    .ease(config.animationEasing);
  
  // Create interactive legend
  const legend = svg.append("g")
    .attr("class", "legend")
    .attr("transform", `translate(${width + 20}, 20)`);
  
  const legendItems = legend.selectAll(".legend-item")
    .data(sources)
    .enter()
    .append("g")
    .attr("class", "legend-item")
    .attr("transform", (d, i) => `translate(0, ${i * 25})`)
    .style("cursor", "pointer")
    .on("click", function(event, d) {
      // Toggle visibility
      const isActive = !d3.select(this).classed("inactive");
      d3.select(this).classed("inactive", isActive);
      
      const opacity = isActive ? 0.2 : 1;
      const legendOpacity = isActive ? 0.5 : 1;
      
      d3.select(this).select("text")
        .style("opacity", legendOpacity);
      
      d3.select(this).select("circle")
        .style("opacity", legendOpacity);
      
      // Update points
      chartArea.selectAll(".point")
        .filter(data => (data.source || data.species) === d)
        .transition()
        .duration(500)
        .style("opacity", opacity);
    });
  
  legendItems.append("circle")
    .attr("r", 6)
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("fill", d => config.colors(d));
  
  legendItems.append("text")
    .attr("x", 15)
    .attr("y", 0)
    .attr("dy", ".35em")
    .attr("font-size", "12px")
    .attr("fill", "#555")
    .text(d => d);
  
  // Add brush for zooming
  const brush = d3.brush()
    .extent([[0, 0], [width, height]])
    .on("end", brushed);
  
  const brushArea = chartArea.append("g")
    .attr("class", "brush")
    .call(brush);
  
  // Add brush reset button
  const resetButton = svg.append("g")
    .attr("class", "reset-button")
    .attr("transform", `translate(${width - 80}, ${height + 40})`)
    .style("cursor", "pointer")
    .on("click", resetZoom);
  
  resetButton.append("rect")
    .attr("width", 80)
    .attr("height", 25)
    .attr("rx", 4)
    .attr("ry", 4)
    .attr("fill", "#4CAF50");
  
  resetButton.append("text")
    .attr("x", 40)
    .attr("y", 12.5)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("fill", "white")
    .attr("font-size", "12px")
    .text("Reset Zoom");
  
  // Add axis selection dropdowns
  const dropdownArea = svg.append("g")
    .attr("class", "dropdown-area")
    .attr("transform", `translate(${width + 20}, 20)`);
  
  // Add axis selection dropdowns
  const xDropdown = dropdownArea.append("select")
    .attr("class", "x-dropdown")
    .on("change", function() {
      xKey = this.value;
      updateVisualization();
    });
  
  const yDropdown = dropdownArea.append("select")
    .attr("class", "y-dropdown")
    .on("change", function() {
      yKey = this.value;
      updateVisualization();
    });
  
  // Populate dropdown options
  xDropdown.selectAll("option")
    .data(numericColumns)
    .enter()
    .append("option")
    .attr("value", d => d)
    .text(d => d);
  
  yDropdown.selectAll("option")
    .data(numericColumns)
    .enter()
    .append("option")
    .attr("value", d => d)
    .text(d => d);
  
  // Add update button
  const updateButton = svg.append("g")
    .attr("class", "update-button")
    .attr("transform", `translate(${width + 20}, ${height + 40})`)
    .style("cursor", "pointer")
    .on("click", updateVisualization);
  
  updateButton.append("rect")
    .attr("width", 80)
    .attr("height", 25)
    .attr("rx", 4)
    .attr("ry", 4)
    .attr("fill", "#4CAF50");
  
  updateButton.append("text")
    .attr("x", 40)
    .attr("y", 12.5)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("fill", "white")
    .attr("font-size", "12px")
    .text("Update");
}
"""
    }
    
    # Return the requested template or a default one if not found
    return templates.get(viz_type, templates.get("scatterplot", "function createVisualization(data, svgElement) {}"))

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
    
    openai.api_key = api_key
    
    # Get model and parameters from environment variables or use defaults
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")
    max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    base_prompt = f"""
    # D3.js Visualization Generation Task

    Create a sophisticated, interactive D3.js version 7 visualization that follows these requirements:

    ## Core Requirements
    1. Create a function named createVisualization(data, svgElement) that:
       - Clears any previous visualization content
       - Sets up responsive SVG with proper viewBox
       - Creates a configuration object with customizable parameters

    2. Implement proper layout with:
       - Configurable margins (top, right, bottom, left)
       - Responsive dimensions based on container size
       - Styled background with rounded corners and subtle shadow
       - Window resize handler to redraw visualization

    3. Process the data by:
       - Grouping by source/category
       - Detecting numeric columns automatically
       - Setting sensible default axes
       - Handling missing or invalid data

    4. Create scales and axes with:
       - Appropriate scale types based on data
       - Padding in domains for visual clarity
       - Grid lines with configurable opacity
       - Formatted and styled axis ticks

    5. Implement core visualization elements:
       - Bars/points/lines with proper attributes
       - Color scales to differentiate data sources
       - Rounded corners and stroke styling
       - Proper spacing between elements

    6. Add basic interactivity:
       - Detailed tooltips on hover showing all data properties
       - Highlighting effects with smooth transitions
       - Click interactions for detailed information
       - Crosshair guides for precise data reading

    7. Implement advanced interactions:
       - Zoom functionality with constraints
       - Brush component for range selection
       - Reset zoom/brush button
       - Axis selection dropdowns

    8. Add animations and transitions:
       - Entrance animations for elements
       - Staggered animations for sequential effects
       - Smooth transitions for all updates
       - Subtle continuous animations (like pulsing)

    9. Create UI components:
       - Interactive legend for toggling visibility
       - Dynamic title that updates with selected axes
       - Axis labels that update dynamically
       - Controls for changing visualization parameters

    10. Ensure accessibility with:
        - ARIA attributes for screen readers
        - Keyboard navigation where appropriate
        - Appropriate color contrasts
        - Descriptive labels for interactive elements

    11. Optimize performance by:
        - Using efficient data binding and updates
        - Implementing clipping paths
        - Handling large datasets appropriately
        - Optimizing animation performance

    12. Handle errors gracefully:
        - Validating input data
        - Managing edge cases
        - Providing fallbacks
        - Including error messages for debugging

    ## Data Format
    The data will be an array of objects, where each object represents a data point with properties. The 'source' property indicates which dataset the point comes from.

    ## Example Data
    ```json
    [
      {{"category": "A", "value": 10, "source": "Dataset 1", "otherValue": 5}},
      {{"category": "B", "value": 15, "source": "Dataset 1", "otherValue": 8}},
      {{"category": "A", "value": 8, "source": "Dataset 2", "otherValue": 12}},
      {{"category": "B", "value": 20, "source": "Dataset 2", "otherValue": 6}}
    ]
    ```

    Your code should be complete, well-commented, and ready to use. Focus on creating a comparative visualization that highlights differences between data sources.

    Data Schema:
    {schema_str}

    Sample Data:
    {json.dumps(data_sample[:5], indent=2)}

    IMPORTANT: Your entire response must be valid D3.js code that can be executed directly. Do not include any text before or after the code.
    """
    
    if user_input:
        prompt = f"""
        # D3.js Visualization Generation Task

        Create a sophisticated, interactive D3.js version 7 visualization that follows these requirements:

        ## Core Requirements
        1. Create a function named createVisualization(data, svgElement) that:
           - Clears any previous visualization content
           - Sets up responsive SVG with proper viewBox
           - Creates a configuration object with customizable parameters

        2. Implement proper layout with:
           - Configurable margins (top, right, bottom, left)
           - Responsive dimensions based on container size
           - Styled background with rounded corners and subtle shadow
           - Window resize handler to redraw visualization

        3. Process the data by:
           - Grouping by source/category
           - Detecting numeric columns automatically
           - Setting sensible default axes
           - Handling missing or invalid data

        4. Create scales and axes with:
           - Appropriate scale types based on data
           - Padding in domains for visual clarity
           - Grid lines with configurable opacity
           - Formatted and styled axis ticks

        5. Implement core visualization elements:
           - Bars/points/lines with proper attributes
           - Color scales to differentiate data sources
           - Rounded corners and stroke styling
           - Proper spacing between elements

        6. Add basic interactivity:
           - Detailed tooltips on hover showing all data properties
           - Highlighting effects with smooth transitions
           - Click interactions for detailed information
           - Crosshair guides for precise data reading

        7. Implement advanced interactions:
           - Zoom functionality with constraints
           - Brush component for range selection
           - Reset zoom/brush button
           - Axis selection dropdowns

        8. Add animations and transitions:
           - Entrance animations for elements
           - Staggered animations for sequential effects
           - Smooth transitions for all updates
           - Subtle continuous animations (like pulsing)

        9. Create UI components:
           - Interactive legend for toggling visibility
           - Dynamic title that updates with selected axes
           - Axis labels that update dynamically
           - Controls for changing visualization parameters

        10. Ensure accessibility with:
            - ARIA attributes for screen readers
            - Keyboard navigation where appropriate
            - Appropriate color contrasts
            - Descriptive labels for interactive elements

        11. Optimize performance by:
            - Using efficient data binding and updates
            - Implementing clipping paths
            - Handling large datasets appropriately
            - Optimizing animation performance

        12. Handle errors gracefully:
            - Validating input data
            - Managing edge cases
            - Providing fallbacks
            - Including error messages for debugging

        ## Data Format
        The data will be an array of objects, where each object represents a data point with properties. The 'source' property indicates which dataset the point comes from.

        ## Example Data
        ```json
        [
          {{"category": "A", "value": 10, "source": "Dataset 1", "otherValue": 5}},
          {{"category": "B", "value": 15, "source": "Dataset 1", "otherValue": 8}},
          {{"category": "A", "value": 8, "source": "Dataset 2", "otherValue": 12}},
          {{"category": "B", "value": 20, "source": "Dataset 2", "otherValue": 6}}
        ]
        ```

        Your code should be complete, well-commented, and ready to use. Focus on creating a comparative visualization that highlights differences between data sources.

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
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
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
    openai.api_key = api_key
    
    # Get model and parameters from environment variables or use defaults
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")
    max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
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
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": refinement_prompt}],
            temperature=temperature,
            max_tokens=max_tokens
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
    Display the D3.js visualization using an iframe and add a download button.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3-legend/2.25.6/d3-legend.min.js"></script>
        <style>
            #visualization {{
                width: 100%;
                height: 100vh;
                overflow: hidden;
            }}
            svg {{
                width: 100%;
                height: 100%;
            }}
            .tooltip {{
                position: absolute;
                background-color: white;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div id="visualization"></div>
        <script>
            {d3_code}
            
            // Create the SVG element
            const svgElement = d3.select("#visualization")
                .append("svg")
                .attr("width", 960)
                .attr("height", 540)
                .attr("viewBox", "0 0 960 540")
                .attr("preserveAspectRatio", "xMidYMid meet");
            
            // Get the data from the parent window
            const vizData = JSON.parse(decodeURIComponent(window.location.hash.slice(1)));
            
            // Call the createVisualization function
            createVisualization(vizData, svgElement);

            // Make the visualization responsive
            window.addEventListener('resize', function() {{
                const width = window.innerWidth;
                const height = window.innerHeight;
                svgElement.attr("width", width).attr("height", height);
                svgElement.attr("viewBox", `0 0 ${{width}} ${{height}}`);
                createVisualization(vizData, svgElement);
            }});
        </script>
    </body>
    </html>
    """
    
    # Encode the data to pass it to the iframe
    encoded_data = urllib.parse.quote(json.dumps(st.session_state.preprocessed_df.to_dict(orient='records')))
    
    # Display the iframe with the encoded data in the URL hash
    st.components.v1.iframe(f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}#{encoded_data}", 
                            width=960, height=540, scrolling=True)

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
    """
    Generate, validate, and if necessary, refine D3 code.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        api_key (str): OpenAI API key.
        user_input (str, optional): Additional user requirements for visualization.
    
    Returns:
        str: Validated D3.js code.
    """
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

    if file1 and file2:
        try:
            if 'preprocessed_df' not in st.session_state or st.session_state.preprocessed_df is None:
                with st.spinner("Preprocessing data..."):
                    merged_df = preprocess_data(file1, file2)
                st.session_state.preprocessed_df = merged_df
            
            with st.expander("Preview of preprocessed data"):
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
                    edit_enabled = st.toggle("Edit", key="edit_toggle")
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
                        st.write(f'<textarea style="position: absolute; left: -9999px;">{code_editor}</textarea>', unsafe_allow_html=True)
                        st.write('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)

            with st.expander("Workflow History"):
                for i, step in enumerate(st.session_state.workflow_history):
                    st.subheader(f"Step {i+1}")
                    st.write(f"Request: {step['request']}")
                    if st.button(f"Revert to Step {i+1}"):
                        st.session_state.current_viz = step['code']
                        # Update the visualization in place
                        with viz_placeholder.container():
                            st.subheader("Current Visualization")
                            display_visualization(st.session_state.current_viz)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main function: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("An unexpected error occurred. Please try again or contact support if the problem persists.")
            st.code(traceback.format_exc())  # Display traceback for debugging
    else:
        st.info("Please upload both CSV files to visualize your data")

if __name__ == "__main__":
    main()
