# AgentRAG Explorer

An intelligent document question-answering system that uses Agentic RAG (Retrieval-Augmented Generation) to provide comprehensive answers to user queries. This system combines the power of large language models with dynamic information retrieval to generate accurate and contextual responses.

## 🌟 Features

- **Intelligent Query Processing**: Breaks down complex queries into simpler sub-queries
- **Dynamic Information Retrieval**: Adaptively retrieves relevant information
- **Multiple Data Sources**:
  - Sample AI/ML documents
  - Wikipedia articles (configurable size and topics)
  - Custom document upload
- **Interactive UI**: Built with Streamlit for easy interaction
- **Performance Metrics**: Track processing time, chunks retrieved, and more
- **Response Validation**: Ensures accuracy and completeness of responses

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- Virtual environment (recommended)
- Hugging Face API token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-rag-explorer.git
cd agentic-rag-explorer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Add your Hugging Face API token to `.env`:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

### Running the Application

```bash
streamlit run src/app.py
```

## 🏗️ Project Structure

```
agentic-rag-explorer/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── src/
│   ├── __init__.py
│   ├── agent_rag.py
│   ├── app.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── evaluation.py
└── cache/
    ├── models/
    └── embeddings/
```

## 🛠️ Usage

1. Start the application
2. Select data source:
   - Sample Documents: Pre-loaded AI/ML content
   - Wikipedia Articles: Choose number of articles and topics
   - Custom Upload: Upload your own documents
3. Initialize the system
4. Enter your questions
5. View detailed analysis of responses

## 📊 Features in Detail

### Query Processing
- Breaks down complex queries into manageable sub-queries
- Dynamically determines when more information is needed
- Generates relevant follow-up questions

### Information Retrieval
- Uses FAISS for efficient similarity search
- Implements semantic search using embeddings
- Removes duplicate information

### Response Generation
- Synthesizes information from multiple sources
- Validates response accuracy
- Provides confidence metrics

## 🔧 Configuration

Key configurations in `.env`:
```
MODEL_NAME=google/flan-t5-base
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## 📝 Examples

Example queries to try:
1. "What is artificial intelligence?"
2. "Compare machine learning and deep learning"
3. "Explain the relationship between AI and neural networks"

## ⚙️ Performance Optimization

- Uses caching for model artifacts
- Implements query response caching
- Optimizes memory usage
- Provides batch processing for large documents

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull request

## 🙏 Acknowledgments

- HuggingFace for providing model access
- Streamlit for the UI framework
- Wikipedia for the dataset
- LangChain for RAG components

## 🔮 Future Improvements

- [ ] Add support for more document formats
- [ ] Implement advanced caching strategies
- [ ] Add visualization for query processing
- [ ] Enhance topic filtering capabilities
- [ ] Implement batch processing for large documents
