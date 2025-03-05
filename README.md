# 🎨 Comparative Visualization Generator

You can play with the [Live Demo](https://comparative-visualization-generator.streamlit.app/) here

## Architecture

```mermaid
flowchart LR
    classDef userClass fill:#f9f,stroke:#333,stroke-width:2px
    classDef frontendClass fill:#bbf,stroke:#33f,stroke-width:1px
    classDef backendClass fill:#bfb,stroke:#3b3,stroke-width:1px
    
    User([User]):::userClass
    
    subgraph Frontend
        direction TB
        SI[Streamlit Interface]:::frontendClass
        VR[Visualization Rendering]:::frontendClass
        HM[History Management]:::frontendClass
    end
    
    subgraph Backend
        direction TB
        DP[Data Preparation]:::backendClass
        OAI[OpenAI API]:::backendClass
        CV[Code Validation & Refinement]:::backendClass
        CSW[Code Safety Wrapper]:::backendClass
    end
    
    User -->|1. API Key| SI
    User -->|2. CSV Files| SI
    User -->|3. Natural Language Query| SI
    
    SI -->|Preprocesses| DP
    DP -->|Merged DataFrame| OAI
    SI -->|Data Sample & Query| OAI
    
    OAI -->|D3.js Code| CV
    CV -->|Invalid| OAI
    CV -->|Valid| CSW
    
    CSW -->|Safe D3.js Code| VR
    VR -->|Interactive Visualization| SI
    
    SI -->|Results| User
    
    HM <-->|Version History| SI
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
