# src/app.py
import os
import streamlit as st
from pathlib import Path
import json
import time
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from utils.data_loader import DocumentLoader
from utils.evaluation import MetricsTracker

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv('HUGGINGFACE_API_TOKEN'):
    st.error("HUGGINGFACE_API_TOKEN not found in environment variables")
    st.info("Please set up your .env file with the required API token")
    st.stop()

# Import your AgentRAG class
try:
    from agent_rag import AgentRAG
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    st.info("Please make sure all required packages are installed and the project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AgentRAG Explorer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = MetricsTracker()

# In src/app.py

def select_data_source():
    """Allow user to select data source"""
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Sample Documents", "Small Wiki Sample", "Custom Upload"]
    )
    
    if data_source == "Sample Documents":
        sample_texts = [
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
            "Machine Learning is a subset of AI that enables systems to learn from data.",
            "Natural Language Processing (NLP) helps computers understand human language.",
            "Deep Learning models are inspired by the human brain's neural networks."
        ]
        return DocumentLoader.convert_to_documents(sample_texts)
    
    elif data_source == "Small Wiki Sample":
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikipedia", "20220301.en", split="train[:10]")
            return DocumentLoader.process_wiki_data(dataset)
        except Exception as e:
            st.error(f"Error loading Wikipedia data: {str(e)}")
            return []
    
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your document", type=['txt'])
        if uploaded_file:
            content = uploaded_file.getvalue().decode()
            return DocumentLoader.process_uploaded_file(content)
        return []

def load_sample_data():
    """Load a small sample dataset for demonstration"""
    try:
        # Option 1: Load just a tiny fraction of Wikipedia
        from datasets import load_dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:10]")
        return dataset['text']
        
        # Option 2: Use a much smaller dataset
        # dataset = load_dataset("squad", split="train[:100]")
        # return [item['context'] for item in dataset]
        
        # Option 3: Return some sample documents
        # return [
        #     "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        #     "Machine Learning is a subset of AI that enables systems to learn from data.",
        #     "Natural Language Processing (NLP) helps computers understand human language.",
        #     "Deep Learning models are inspired by the human brain's neural networks."
        # ]
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return []

def initialize_agent():
    """Initialize the AgentRAG system"""
    if st.session_state.agent is None:
        with st.spinner("Initializing AgentRAG... This might take a minute."):
            try:
                # Get data source
                documents = select_data_source()
                if not documents:
                    st.warning("Please select or upload documents to continue.")
                    return
                
                st.session_state.agent = AgentRAG()
                st.session_state.agent.load_and_index_data(documents)
                st.success("AgentRAG initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing AgentRAG: {str(e)}")

def display_metrics(metrics: Dict):
    """Display query processing metrics"""
    cols = st.columns(4)
    with cols[0]:
        st.metric("Processing Time", f"{metrics['processing_time']:.2f}s")
    with cols[1]:
        st.metric("Sub-queries", metrics['num_sub_queries'])
    with cols[2]:
        st.metric("Chunks Retrieved", metrics['num_chunks'])
    with cols[3]:
        st.metric("Memory Usage (MB)", metrics.get('memory_usage', 0))


def display_response(response_data: Dict, in_history: bool = False):
    """
    Display the query response and details.
    
    Args:
        response_data: The response data to display
        in_history: Whether this is being displayed in the history section
    """
    # Error handling
    if 'error' in response_data:
        st.error(response_data['error'])
        return

    # Display main response in a clean format
    st.markdown("### Answer")
    if isinstance(response_data, str):
        st.write(response_data)
    else:
        st.markdown(f"_{response_data.get('response', 'No response generated')}_")
    
    # Show details in expander
    with st.expander("🔍 View Detailed Analysis", expanded=False):
        # Create columns for metrics
        if 'metrics' in response_data:
            metrics = response_data['metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{metrics.get('processing_time', 0):.2f}s")
            with col2:
                st.metric("Sub-queries", metrics.get('num_sub_queries', 0))
            with col3:
                st.metric("Chunks Retrieved", metrics.get('num_chunks', 0))
        
        # Display thought process
        st.markdown("#### 🤔 Thought Process")
        st.markdown("The question was broken down into these components:")
        sub_queries = response_data.get('sub_queries', [])
        for i, query in enumerate(sub_queries, 1):
            st.markdown(f"- {query}")
        
        # Display relevant information
        st.markdown("#### 📚 Relevant Information")
        chunks = response_data.get('chunks', [])
        all_chunks = []
        
        # Collect and clean chunk content
        for chunk in chunks:
            if hasattr(chunk, 'page_content'):
                content = chunk.page_content
            else:
                content = str(chunk)
            # Clean up the content
            content = content.strip()
            if content:
                all_chunks.append(content)
        
        # Display consolidated information
        if all_chunks:
            consolidated_info = " ".join(all_chunks)
            st.markdown(consolidated_info)
        else:
            st.info("No relevant chunks were found.")
        
        # Show validation status if available
        if 'metrics' in response_data and 'validation_passed' in response_data['metrics']:
            st.markdown("#### ✅ Validation")
            validation_status = "Passed" if response_data['metrics']['validation_passed'] else "Failed"
            status_color = "green" if response_data['metrics']['validation_passed'] else "red"
            st.markdown(f"Response Validation: :{status_color}[{validation_status}]")

    # Add a divider if we're displaying multiple responses
    if not in_history:
        st.markdown("---")

def display_system_stats(agent):
    """Display system statistics in a formatted way."""
    try:
        stats = agent.get_stats()
        st.sidebar.markdown("### System Statistics")
        st.sidebar.markdown(f"""
        - Documents indexed: {stats['num_documents']}
        - Queries cached: {stats['num_cached_queries']}
        - Chunk size: {stats['chunk_size']}
        - Chunk overlap: {stats['chunk_overlap']}
        """)
        
        with st.sidebar.expander("Model Details", expanded=False):
            st.write(f"Embedding model: {stats['embedding_model']}")
            st.write(f"LLM model: {stats['llm_model']}")
    except Exception as e:
        st.sidebar.warning(f"Could not load system stats: {str(e)}")

def main():
    # Initialize session state
    init_session_state()
    
    # Page header
    st.title("AgentRAG Explorer 🤖")
    st.markdown("""
    This demo showcases Agentic RAG (Retrieval-Augmented Generation) using:
    - 🧠 Intelligent query planning
    - 📚 Dynamic information retrieval
    - 🔄 Iterative refinement
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        if st.button("Initialize System"):
            initialize_agent()
        
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.agent is not None:
            st.success("System Ready")
            # Display system stats using the new function
            display_system_stats(st.session_state.agent)
        else:
            st.warning("System Not Initialized")
        
        st.markdown("---")
        if st.button("Clear History"):
            st.session_state.history = []
    
    # Main content
    if st.session_state.agent is not None:
        # Query input
        query = st.text_input("Enter your question:")
        col1, col2 = st.columns([1, 5])
        with col1:
            process_query = st.button("Submit")
        
        if process_query and query:
            with st.spinner("Processing query..."):
                try:
                    # Process query
                    response = st.session_state.agent.query(query)
                    
                    # Store in history
                    st.session_state.history.append({
                        'query': query,
                        'response': response,
                        'timestamp': time.time()
                    })
                    
                    # Display results
                    if 'metrics' in response:
                        display_metrics(response['metrics'])
                    display_response(response)
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        # Display history
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### Query History")
            for item in reversed(st.session_state.history):
                with st.expander(f"Q: {item['query']}", expanded=False):
                    st.write("Query:", item['query'])
                    st.write("Response:", item['response'].get('response', ''))
                    st.write("Timestamp:", time.strftime('%Y-%m-%d %H:%M:%S', 
                                                       time.localtime(item['timestamp'])))
    else:
        st.info("Please initialize the system using the sidebar button.")

if __name__ == "__main__":
    main()
