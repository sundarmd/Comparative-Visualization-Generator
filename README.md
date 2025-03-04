# 🎨 Comparative Visualization Generator

You can play with the [Live Demo](https://comparative-visualization-generator.streamlit.app/) here

## Architecture

```mermaid
graph TD
    A[User] -->|1. Provides API Key| B[Streamlit Interface]
    A -->|2. Uploads CSV Files| B
    A -->|3. Enters Natural Language Query| B
    
    B -->|Preprocesses Data| C[Data Preparation]
    C -->|Merged DataFrame| D[OpenAI API]
    
    B -->|Sends Prompt with Data Sample & Query| D
    D -->|Returns D3.js Code| E[Code Validation & Refinement]
    
    E -->|Invalid Code| D
    E -->|Valid Code| F[Code Safety Wrapper]
    
    F -->|Safe D3.js Code| G[Visualization Rendering]
    G -->|Interactive Visualization| B
    
    B -->|Displays Results| A
    
    H[History Management] <-->|Stores Previous Versions| B
    
    subgraph Backend Processing
        C
        D
        E
        F
    end
    
    subgraph Frontend
        B
        G
        H
    end
```

## How it works?

1. Enter your Open AI API key

![image](https://github.com/user-attachments/assets/2c155ed8-7baf-47fb-af20-f008433a6453)

2. Upload 2 source files ( .csv ) with the same schema

![image](https://github.com/user-attachments/assets/ef1434bc-f8c3-4292-8746-8cc5972f9fbf)

3. Iteratively interact to re-generate visualizations based on the data through natural language queries

![image](https://github.com/user-attachments/assets/9e066deb-4134-474d-b411-43b45bd7bf86)

![image](https://github.com/user-attachments/assets/680bbde2-f91e-43d5-9b45-bdf4f8d94be4)

![image](https://github.com/user-attachments/assets/4ec8041b-8471-4933-9f56-71c1e3bb2aee)

![image](https://github.com/user-attachments/assets/6e544d91-8a5b-4a70-b944-e18e9c6de149)


## Features

1. The nature of visualization will be comparative always
2. The visualizations are interactive because they are generated in D3.js
3. The user needs no coding experience to generate these visualizations
