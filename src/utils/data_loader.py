# src/utils/data_loader.py
from typing import List
from langchain.schema import Document

class DocumentLoader:
    @staticmethod
    def convert_to_documents(texts: List[str]) -> List[Document]:
        """Convert text strings to Document objects"""
        return [
            Document(
                page_content=text,
                metadata={"source": f"document_{i}"}
            )
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def process_wiki_data(dataset) -> List[Document]:
        """Process Wikipedia dataset into documents"""
        texts = dataset['text']
        return DocumentLoader.convert_to_documents(texts)

    @staticmethod
    def process_uploaded_file(file_content: str) -> List[Document]:
        """Process uploaded file content into documents"""
        return DocumentLoader.convert_to_documents([file_content])
