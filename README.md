# 🎨 Comparative Visualization Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://comparative-visualization-generator.streamlit.app/) 
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![D3.js](https://img.shields.io/badge/D3.js-7.x-orange.svg)](https://d3js.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)

Generate interactive, comparative D3.js visualizations through natural language - no coding required!

## ✨ Features

- **📊 Comparative Visualizations** - Automatically compare data from two sources
- **🔄 Interactive D3.js** - All visualizations are interactive and web-ready
- **💬 Natural Language Interface** - Describe what you want in plain English
- **🔄 Iterative Refinement** - Continuously improve visualizations through conversation
- **📋 No Coding Required** - Technical complexity handled behind the scenes

## 🏗️ Architecture

```mermaid
flowchart LR
    classDef userClass fill:#f9f,stroke:#333,stroke-width:2px,color:#333,font-weight:bold
    classDef frontendClass fill:#bbf,stroke:#33f,stroke-width:1px,color:#005
    classDef backendClass fill:#bfb,stroke:#3b3,stroke-width:1px,color:#050
    classDef dataClass fill:#ffd,stroke:#b90,stroke-width:1px,color:#840
    
    User([👤 User]):::userClass
    
    subgraph Frontend["🖥️ Frontend"]
        direction TB
        SI[Streamlit Interface]:::frontendClass
        VR[Visualization Renderer]:::frontendClass
        HM[History Manager]:::frontendClass
    end
    
    subgraph Backend["⚙️ Backend"]
        direction TB
        DP[Data Processor]:::backendClass
        OAI[OpenAI API]:::backendClass
        CV[Code Validator]:::backendClass
        CSW[Safety Wrapper]:::backendClass
    end
    
    User -->|"1️⃣ API Key"| SI
    User -->|"2️⃣ CSV Files"| SI
    User -->|"3️⃣ Natural Language"| SI
    
    SI -->|Preprocess| DP
    DP -->|Structured Data| OAI
    SI -->|Query Context| OAI
    
    OAI -->|D3.js Code| CV
    CV -->|Invalid| OAI
    CV -->|Valid| CSW
    
    CSW -->|Safe Code| VR
    VR -->|Visualization| SI
    
    SI -->|Results| User
    
    HM <-->|Version History| SI

    style Frontend fill:#eef,stroke:#99c,stroke-width:2px
    style Backend fill:#efe,stroke:#9c9,stroke-width:2px
```

## 🚀 How It Works

### 1️⃣ Enter your OpenAI API key
Connect to the OpenAI API to power the visualization generation engine.

![API Key Entry](https://github.com/user-attachments/assets/2c155ed8-7baf-47fb-af20-f008433a6453)

### 2️⃣ Upload your data files
Provide two CSV files with matching schemas for comparison.

![File Upload](https://github.com/user-attachments/assets/ef1434bc-f8c3-4292-8746-8cc5972f9fbf)

### 3️⃣ Create visualizations through conversation
Simply describe what you'd like to see in natural language.

![Natural Language Query](https://github.com/user-attachments/assets/9e066deb-4134-474d-b411-43b45bd7bf86)

### 4️⃣ Interact with your visualizations
Explore the data through interactive D3.js visualizations.

![Interactive Visualization](https://github.com/user-attachments/assets/680bbde2-f91e-43d5-9b45-bdf4f8d94be4)

![Visualization Example](https://github.com/user-attachments/assets/4ec8041b-8471-4933-9f56-71c1e3bb2aee)

![Another Example](https://github.com/user-attachments/assets/6e544d91-8a5b-4a70-b944-e18e9c6de149)

## 📋 Requirements

- Python 3.7+
- Streamlit
- OpenAI API key
- Internet connection for D3.js libraries

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/comparative-visualization-generator.git

# Navigate to project directory
cd comparative-visualization-generator

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
