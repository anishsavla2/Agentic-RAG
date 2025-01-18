# src/agent_rag.py
import os
import torch
import gc
import time
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

class AgentRAG:
    def __init__(self, 
                model_name: str = None,
                embedding_model: str = None,
                chunk_size: int = None,
                chunk_overlap: int = None):
        """Initialize the AgentRAG system."""
        # Load settings from environment variables or use defaults
        self.model_name = model_name or os.getenv('MODEL_NAME', "google/flan-t5-base")
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 
                                                          "sentence-transformers/all-mpnet-base-v2")
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', 50))
        
        # Get API token
        self.huggingface_api_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if not self.huggingface_api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        self.llm = HuggingFaceHub(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.huggingface_api_token,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "top_p": 0.95
            }
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Initialize metrics tracking
        self.last_metrics = {}
        self.last_sub_queries = []
        self.last_chunks = []
        
        # Initialize cache
        self._query_cache = {}

    def _clean_memory(self):
        """Clean up memory after processing."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_cached_response(self, query: str) -> Optional[Dict]:
        """Get cached response for a query if available."""
        return self._query_cache.get(query)

    def _cache_response(self, query: str, response: Dict):
        """Cache the response for a query."""
        self._query_cache[query] = response

    def load_and_index_data(self, documents: List[Document]) -> bool:
        """
        Load and index documents into the vector store.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            chunks = self.text_splitter.split_documents(documents)
            
            if not self.vector_store:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            
            processing_time = time.time() - start_time
            print(f"Indexed {len(chunks)} chunks in {processing_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error in load_and_index_data: {str(e)}")
            return False
        finally:
            self._clean_memory()

    def retrieve_relevant_info(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: The query to search for
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return []

    def plan_retrieval(self, query: str) -> List[str]:
        """
        Generate sub-queries for complex queries.
        
        Args:
            query: The original query
            
        Returns:
            List of sub-queries
        """
        planning_prompt = PromptTemplate(
            template="""Break down this query into simple sub-queries for information retrieval:
            Query: {query}
            
            Return only the sub-queries, one per line. Keep them simple and focused.""",
            input_variables=["query"]
        )
        
        planning_chain = LLMChain(llm=self.llm, prompt=planning_prompt)
        result = planning_chain.run(query=query)
        
        sub_queries = [q.strip() for q in result.split('\n') if q.strip()]
        self.last_sub_queries = sub_queries
        return sub_queries

    def needs_more_info(self, chunks: List[Document], query: str) -> bool:
        """
        Determine if more information is needed.
        
        Args:
            chunks: Currently retrieved chunks
            query: The query being processed
            
        Returns:
            Boolean indicating if more information is needed
        """
        if not chunks:
            return True
            
        confidence_prompt = PromptTemplate(
            template="""Given the query: {query}
            And the following information:
            {chunks}
            
            Do we have enough information to provide a complete answer? 
            Return only YES or NO.""",
            input_variables=["query", "chunks"]
        )
        
        confidence_chain = LLMChain(llm=self.llm, prompt=confidence_prompt)
        chunks_text = "\n".join([chunk.page_content for chunk in chunks])
        result = confidence_chain.run(query=query, chunks=chunks_text)
        
        return "NO" in result.upper()

    def generate_follow_up_queries(self, query: str, chunks: List[Document]) -> List[str]:
        """Generate follow-up queries based on missing information."""
        followup_prompt = PromptTemplate(
            template="""Given the original query: {query}
            And the current information we have:
            {chunks}
            
            What follow-up questions should we ask to get more complete information?
            Return only the questions, one per line.""",
            input_variables=["query", "chunks"]
        )
        
        followup_chain = LLMChain(llm=self.llm, prompt=followup_prompt)
        chunks_text = "\n".join([chunk.page_content for chunk in chunks])
        result = followup_chain.run(query=query, chunks=chunks_text)
        
        return [q.strip() for q in result.split('\n') if q.strip()]

    # In agent_rag.py, update the synthesize_information method:

    def synthesize_information(self, query: str, chunks: List[Document]) -> str:
        """
        Synthesize collected information into a coherent response.
        
        Args:
            query: Original query
            chunks: All relevant chunks
            
        Returns:
            Synthesized response
        """
        synthesis_prompt = PromptTemplate(
            template="""Based on the following information:
            {chunks}
            
            Please provide a comprehensive and well-organized answer to this question:
            {query}
            
            Instructions:
            1. Use the provided information to create a clear and direct answer
            2. Focus on accuracy and completeness
            3. Present the information in a logical flow
            4. Use simple, clear language
            
            Answer:""",
            input_variables=["query", "chunks"]
        )
        
        synthesis_chain = LLMChain(llm=self.llm, prompt=synthesis_prompt)
        chunks_text = "\n".join([
            chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            for chunk in chunks
        ])
        
        try:
            response = synthesis_chain.run(
                query=query,
                chunks=chunks_text
            )
            
            # Ensure we have a response
            if not response or response.strip() == "":
                # Fallback to direct chunk if synthesis fails
                return "Based on the available information: " + chunks_text
                
            return response.strip()
        except Exception as e:
            print(f"Error in synthesis: {str(e)}")
            # Return a simple combination of chunks if synthesis fails
            return "Based on the available information: " + chunks_text

    def validate_response(self, response: str, query: str, chunks: List[Document]) -> bool:
        """
        Validate the generated response for factual accuracy.
        
        Args:
            response: Generated response
            query: Original query
            chunks: Source chunks
            
        Returns:
            Boolean indicating if response is valid
        """
        validation_prompt = PromptTemplate(
            template="""Validate this response:
            Query: {query}
            Response: {response}
            Source Information: {chunks}
            
            Is the response:
            1. Accurate according to the sources?
            2. Complete in answering the query?
            3. Well-supported by the provided information?
            
            Return only VALID or INVALID.""",
            input_variables=["query", "response", "chunks"]
        )
        
        validation_chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        chunks_text = "\n".join([chunk.page_content for chunk in chunks])
        
        try:
            result = validation_chain.run(
                query=query,
                response=response,
                chunks=chunks_text
            )
            return "VALID" in result.upper()
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return False

    # In agent_rag.py, update the query method:

    def query(self, user_query: str) -> Dict:
        """
        Process a user query and return results with metrics.
        
        Args:
            user_query: The user's question
            
        Returns:
            Dict containing response and metrics
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = self._get_cached_response(user_query)
            if cached_response:
                return cached_response

            # Reset tracking
            self.last_chunks = []
            
            # Get sub-queries
            sub_queries = self.plan_retrieval(user_query)
            if not sub_queries:
                sub_queries = [user_query]  # Use original query if no sub-queries generated
            
            # Collect chunks
            all_chunks = []
            for sub_query in sub_queries:
                chunks = self.retrieve_relevant_info(sub_query)
                if chunks:
                    all_chunks.extend(chunks)
            
            # If no chunks found, return appropriate message
            if not all_chunks:
                return {
                    'response': "I apologize, but I couldn't find relevant information to answer your question.",
                    'metrics': {
                        'processing_time': time.time() - start_time,
                        'num_sub_queries': len(sub_queries),
                        'num_chunks': 0
                    },
                    'sub_queries': sub_queries,
                    'chunks': []
                }
            
            # Generate and validate response
            response = self.synthesize_information(user_query, all_chunks)
            
            result = {
                'response': response,
                'metrics': {
                    'processing_time': time.time() - start_time,
                    'num_sub_queries': len(sub_queries),
                    'num_chunks': len(all_chunks)
                },
                'sub_queries': sub_queries,
                'chunks': all_chunks
            }
            
            # Cache the result
            self._cache_response(user_query, result)
            return result
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error while processing your query.",
                'error': str(e),
                'metrics': {
                    'processing_time': time.time() - start_time,
                    'num_sub_queries': 0,
                    'num_chunks': 0
                }
            }

    def get_stats(self) -> Dict:
        """Get system statistics."""
        num_docs = 0
        if self.vector_store:
            try:
                # Get number of vectors in the FAISS index
                num_docs = self.vector_store.index.ntotal
            except AttributeError:
                # Fallback if ntotal is not available
                try:
                    num_docs = len(self.vector_store.docstore._dict)
                except:
                    num_docs = 0
        
        return {
            'num_documents': num_docs,
            'num_cached_queries': len(self._query_cache),
            'embedding_model': self.embedding_model,
            'llm_model': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
