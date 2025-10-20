import os
import fitz  # PyMuPDF - much faster than PyPDF2
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF (100x faster than PyPDF2)."""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            for page_num, page in enumerate(doc):
                extracted_text = page.get_text()
                if extracted_text.strip():  # Only add non-empty text
                    text += extracted_text + "\n"
                
                # Progress indicator every 50 pages
                if (page_num + 1) % 50 == 0:
                    print(f"  Processed {page_num + 1}/{total_pages} pages...")
            
            doc.close()
            print(f"  Extracted text from {total_pages} pages")

        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return ""

        return text

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from a text file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error processing text file {txt_path}: {str(e)}")
            return ""

    def process_document(self, file_path: str, source_name: str = None) -> List[Document]:
        """Process a single document and return chunks."""
        if source_name is None:
            source_name = os.path.basename(file_path)

        # Determine file type and extract text
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {file_ext}")
            return []

        if not text.strip():
            print(f"No text extracted from {file_path}")
            return []

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "file_type": file_ext
                }
            )
            documents.append(doc)

        return documents

    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported documents in a directory."""
        all_documents = []

        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist")
            return []

        supported_extensions = ['.pdf', '.txt']

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()

                if file_ext in supported_extensions:
                    print(f"Processing {filename}...")
                    documents = self.process_document(file_path, filename)
                    all_documents.extend(documents)
                else:
                    print(f"Skipping unsupported file: {filename}")

        print(f"Processed {len(all_documents)} chunks from documents")
        return all_documents

    def get_document_stats(self, documents: List[Document]) -> Dict:
        """Get statistics about processed documents."""
        if not documents:
            return {}

        sources = set()
        total_chars = 0

        for doc in documents:
            sources.add(doc.metadata["source"])
            total_chars += len(doc.page_content)

        return {
            "total_chunks": len(documents),
            "total_documents": len(sources),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(documents) if documents else 0,
            "sources": list(sources)
        }

if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()

    # Process the data directory
    data_dir = "data"
    documents = processor.process_directory(data_dir)

    if documents:
        stats = processor.get_document_stats(documents)
        print("\nDocument Processing Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # Print a sample chunk
        if documents:
            print(f"\nSample chunk from {documents[0].metadata['source']}:")
            print(documents[0].page_content[:200] + "...")
    else:
        print("No documents processed")