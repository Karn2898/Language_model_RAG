#  A Language Model using GPT-2 architecture



A Python-based Retrieval-Augmented Generation system that enhances language model responses by retrieving relevant information from external knowledge sources.
<img width="2068" height="1592" alt="NoteGPT-Sequence Diagram-1769025967483" src="https://github.com/user-attachments/assets/1a06d79a-9284-44c1-a7f3-fd0f44cb98f1" />


## Features

- Custom model architecture for RAG implementation
- Language model wrapper for flexible LLM integration
- Document retrieval and embedding functionality
- Context-aware response generation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Language_model_RAG
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Required packages:
- torch
- numpy
- transformers
- datasets
- accelerate

## Usage

Run the main wrapper script:
```bash
python lang_wrapper.py
```

## Project Structure

```text
Language_model_RAG/


├── model_architecture.py  # Core RAG model implementation
├── lang_wrapper.py        # Language model interface
├── requirements.txt       # Project dependencies
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
