"""
Document Processing and Vectorization System
===========================================

This application provides a complete pipeline for converting raw PDF documents
into vectorized data that can be stored in a vector database for AI retrieval.

Features:
- PDF upload via web interface
- Text extraction from PDFs
- Intelligent text chunking
- Vector embedding generation
- Vector storage in ChromaDB
- RESTful API endpoints

Usage:
1. Set up environment variables (GOOGLE_API_KEY)
2. Install dependencies: pip install -r requirements.txt
3. Run the application: uvicorn main:app --reload
4. Upload PDFs via POST /upload-pdf/ endpoint
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import pypdf  # Alternative to PyMuPDF
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Vectorization System",
    description="Convert PDF documents into vectorized data for AI retrieval",
    version="1.0.0"
)

# Configuration
class Config:
    """Application configuration"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./vector_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set. Embedding generation will fail.")
        return True

# Initialize configuration
Config.validate()

# Configure Google AI if API key is available
if Config.GOOGLE_API_KEY:
    genai.configure(api_key=Config.GOOGLE_API_KEY)

# Initialize ChromaDB
try:
    client = chromadb.PersistentClient(path=Config.CHROMADB_PATH)
    collection = client.get_or_create_collection(name=Config.COLLECTION_NAME)
    logger.info(f"ChromaDB initialized at {Config.CHROMADB_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    collection = None

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)


class DocumentProcessor:
    """Main document processing pipeline"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF using pypdf
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            full_text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {str(e)}")
    
    @staticmethod
    def chunk_text(text: str) -> List[str]:
        """
        Split text into chunks using LangChain's text splitter
        
        Args:
            text: Long text to be chunked
            
        Returns:
            List of text chunks
        """
        try:
            chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise HTTPException(status_code=500, detail=f"Text chunking failed: {str(e)}")
    
    @staticmethod
    def generate_embeddings(chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using Google's embedding model
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        if not Config.GOOGLE_API_KEY:
            raise HTTPException(
                status_code=500, 
                detail="Google API key not configured. Cannot generate embeddings."
            )
        
        try:
            embeddings = []
            
            # Process chunks in batches to avoid API limits
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                # Handle both single and batch embeddings
                if isinstance(result['embedding'][0], list):
                    embeddings.extend(result['embedding'])
                else:
                    embeddings.append(result['embedding'])
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
    @staticmethod
    def store_vectors(chunks: List[str], embeddings: List[List[float]], document_id: str, document_name: str = None) -> Dict:
        """
        Store vectors and text in ChromaDB
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            document_id: Unique identifier for the document
            document_name: Human-readable name for the document
            
        Returns:
            Storage result information
        """
        if not collection:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        try:
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Prepare metadata
            metadatas = [{"document_id": document_id, "chunk_index": i, "document_name": document_name or "Unnamed Document"} for i in range(len(chunks))]
            
            # Store in ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            result = {
                "document_id": document_id,
                "chunks_stored": len(chunks),
                "embeddings_stored": len(embeddings)
            }
            
            logger.info(f"Stored {len(chunks)} vectors for document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")
    
    @classmethod
    def process_pdf(cls, pdf_path: str, document_id: str, document_name: str = None) -> Dict:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            document_id: Unique document identifier
            document_name: Human-readable name for the document
            
        Returns:
            Processing result summary
        """
        try:
            # Step 1: Extract text from PDF
            text = cls.extract_text_from_pdf(pdf_path)
            
            # Step 2: Chunk the text
            chunks = cls.chunk_text(text)
            
            # Step 3: Generate embeddings
            embeddings = cls.generate_embeddings(chunks)
            
            # Step 4: Store vectors
            storage_result = cls.store_vectors(chunks, embeddings, document_id, document_name)
            
            # Cleanup temporary file
            try:
                os.remove(pdf_path)
            except:
                pass
            
            return {
                "status": "success",
                "document_id": document_id,
                "document_name": document_name or "Unnamed Document",
                "text_length": len(text),
                "chunks_created": len(chunks),
                "embeddings_generated": len(embeddings),
                "storage_result": storage_result
            }
            
        except Exception as e:
            # Cleanup on error
            try:
                os.remove(pdf_path)
            except:
                pass
            raise e


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML upload interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Vectorization System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .button:hover { background-color: #45a049; }
            .status { margin: 20px 0; padding: 10px; border-radius: 4px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>Document Vectorization System</h1>
        <p>Upload PDF documents to convert them into vectorized data for AI retrieval.</p>
        
        <div class="upload-box">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" accept=".pdf" required>
                <br><br>
                <input type="text" id="documentName" placeholder="Enter document name (optional)" style="padding: 8px; width: 300px; margin: 10px; border: 1px solid #ccc; border-radius: 4px;">
                <br><br>
                <button type="submit" class="button">Upload and Process PDF</button>
            </form>
        </div>
        
        <div id="status"></div>
        
        <div style="text-align: center; margin: 2rem 0;">
            <a href="/chat/" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; border-radius: 10px; text-decoration: none; font-weight: bold; display: inline-block; transition: transform 0.2s;" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                🤖 Ask Questions About Your Documents
            </a>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const documentNameInput = document.getElementById('documentName');
                const statusDiv = document.getElementById('status');
                
                if (!fileInput.files[0]) {
                    statusDiv.innerHTML = '<div class="error">Please select a file</div>';
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                if (documentNameInput.value.trim()) {
                    formData.append('document_name', documentNameInput.value.trim());
                }
                
                statusDiv.innerHTML = '<div>Processing... This may take a few minutes.</div>';
                
                try {
                    const response = await fetch('/upload-pdf/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        statusDiv.innerHTML = `
                            <div class="success">
                                <h3>Success!</h3>
                                <p>Document Name: ${result.document_name}</p>
                                <p>Document ID: ${result.document_id}</p>
                                <p>Text Length: ${result.text_length} characters</p>
                                <p>Chunks Created: ${result.chunks_created}</p>
                                <p>Embeddings Generated: ${result.embeddings_generated}</p>
                            </div>`;
                    } else {
                        statusDiv.innerHTML = `<div class="error">Error: ${result.detail}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), document_name: Optional[str] = None):
    """
    Upload and process a PDF document with optional custom name
    
    Args:
        file: PDF file to process
        document_name: Optional custom name for the document (defaults to filename)
        
    Returns:
        Processing result with document info
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    # Use custom name if provided, otherwise use filename
    display_name = document_name.strip() if document_name else file.filename
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{document_id}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF (this runs synchronously for now)
        result = DocumentProcessor.process_pdf(temp_path, document_id, display_name)
        
        return result
        
    except Exception as e:
        # Cleanup on error
        try:
            os.remove(temp_path)
        except:
            pass
        raise e


@app.get("/search/")
async def search_documents(query: str, limit: int = 5):
    """
    Search for similar documents using vector similarity
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of similar document chunks
    """
    if not Config.GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    if not collection:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    try:
        # Generate embedding for the query
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = result['embedding']
        
        # Search for similar documents
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        # Format results
        formatted_results = []
        for i in range(len(search_results['documents'][0])):
            formatted_results.append({
                "document": search_results['documents'][0][i],
                "metadata": search_results['metadatas'][0][i],
                "distance": search_results['distances'][0][i]
            })
        
        return {
            "query": query,
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ask/")
async def ask_question(query: str, document_id: Optional[str] = None, limit: int = 5):
    """
    Ask a question and get an AI-generated answer based on document content
    
    Args:
        query: Question to ask
        document_id: Optional specific document ID to search in
        limit: Maximum number of context chunks to use
        
    Returns:
        AI-generated answer with sources
    """
    if not Config.GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    if not collection:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    try:
        # Generate embedding for the query
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = result['embedding']
        
        # Build search filter for specific document if provided
        where_filter = {}
        if document_id:
            where_filter = {"document_id": document_id}
        
        # Search for similar documents
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter if where_filter else None
        )
        
        if not search_results['documents'][0]:
            return {
                "query": query,
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "document_id": document_id
            }
        
        # Combine top results as context
        context_chunks = search_results['documents'][0]
        context = "\n\n".join(context_chunks)
        
        # Generate answer using Gemini with improved prompt
        prompt = f"""You are a helpful assistant that provides specific, practical, and actionable answers based on the given context.

INSTRUCTIONS:
- Use the CONTEXT below to answer the QUESTION
- Give specific examples, recommendations, or steps when possible
- If the context mentions general concepts, provide concrete practical applications
- For health/exercise questions, suggest specific activities or routines
- For technical questions, provide step-by-step guidance
- If you don't know the answer based on the context, say so clearly
- Be conversational but informative

CONTEXT: {context}

QUESTION: {query}

Please provide a helpful, specific answer with practical recommendations based on the context above:"""

        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        
        # Format sources
        sources = []
        for i, doc in enumerate(context_chunks):
            metadata = search_results['metadatas'][0][i] if i < len(search_results['metadatas'][0]) else {}
            sources.append({
                "chunk_index": i,
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "metadata": metadata,
                "relevance_score": 1 - search_results['distances'][0][i] if i < len(search_results['distances'][0]) else 0
            })
        
        return {
            "query": query,
            "answer": response.text,
            "sources": sources,
            "document_id": document_id,
            "total_sources": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.get("/chat/")
async def chat_interface():
    """Serve a chat interface for asking questions about documents"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Q&A System</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                text-align: center;
            }
            .chat-container {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
            .query-form {
                display: flex;
                gap: 10px;
                margin-bottom: 2rem;
            }
            .query-input {
                flex: 1;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            .query-input:focus {
                border-color: #667eea;
            }
            .ask-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .ask-button:hover {
                transform: translateY(-2px);
            }
            .document-filter {
                margin-bottom: 1rem;
            }
            .document-select {
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                width: 300px;
            }
            .document-management {
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            .delete-button {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background-color 0.3s;
            }
            .delete-button:hover {
                background-color: #c82333;
            }
            .delete-button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .refresh-button {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }
            .refresh-button:hover {
                background-color: #218838;
            }
            .answer-container {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 0 10px 10px 0;
            }
            .question {
                background: #e3f2fd;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 4px solid #2196f3;
            }
            .sources {
                margin-top: 1.5rem;
                background: #fff3e0;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #ff9800;
            }
            .source-item {
                background: white;
                margin: 0.5rem 0;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 2rem;
                color: #667eea;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .example-questions {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .example-btn {
                background: #e3f2fd;
                border: 1px solid #2196f3;
                color: #1976d2;
                padding: 8px 16px;
                margin: 4px;
                border-radius: 20px;
                cursor: pointer;
                display: inline-block;
                transition: all 0.2s;
            }
            .example-btn:hover {
                background: #2196f3;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Document Q&A System</h1>
            <p>Ask questions about your uploaded PDF documents and get AI-powered answers</p>
        </div>
        
        <div class="chat-container">
            <div class="document-filter">
                <label for="documentSelect">Filter by Document (optional):</label>
                <div class="document-management">
                    <select id="documentSelect" class="document-select">
                        <option value="">All Documents</option>
                    </select>
                    <button id="refreshButton" onclick="loadDocuments()" class="refresh-button">🔄 Refresh</button>
                    <button id="deleteButton" onclick="deleteSelectedDocument()" class="delete-button" disabled>🗑️ Delete Selected</button>
                </div>
            </div>
            
            <div class="query-form">
                <input type="text" id="queryInput" class="query-input" 
                       placeholder="Ask a question about your documents..." 
                       onkeypress="if(event.key==='Enter') askQuestion()">
                <button onclick="askQuestion()" class="ask-button">Ask Question</button>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Searching documents and generating answer...</p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <div class="example-questions">
            <h3>💡 Example Questions:</h3>
            <div class="example-btn" onclick="setQuery('What is the main topic of this document?')">
                What is the main topic of this document?
            </div>
            <div class="example-btn" onclick="setQuery('Can you summarize the key points?')">
                Can you summarize the key points?
            </div>
            <div class="example-btn" onclick="setQuery('What are the main conclusions?')">
                What are the main conclusions?
            </div>
            <div class="example-btn" onclick="setQuery('Tell me about the methodology used')">
                Tell me about the methodology used
            </div>
        </div>
        
        <script>
            function setQuery(query) {
                document.getElementById('queryInput').value = query;
            }
            
            async function loadDocuments() {
                try {
                    const response = await fetch('/documents/');
                    const data = await response.json();
                    
                    const select = document.getElementById('documentSelect');
                    select.innerHTML = '<option value="">All Documents (' + data.total_chunks + ' chunks)</option>';
                    
                    if (data.documents && data.documents.length > 0) {
                        data.documents.forEach(doc => {
                            const option = document.createElement('option');
                            option.value = doc.document_id;
                            option.textContent = doc.document_name + ' (' + doc.chunk_count + ' chunks)';
                            select.appendChild(option);
                        });
                    }
                    
                    // Update delete button state
                    updateDeleteButtonState();
                } catch (error) {
                    console.error('Error loading documents:', error);
                    const select = document.getElementById('documentSelect');
                    select.innerHTML = '<option value="">All Documents (Error loading)</option>';
                    updateDeleteButtonState();
                }
            }
            
            function updateDeleteButtonState() {
                const select = document.getElementById('documentSelect');
                const deleteButton = document.getElementById('deleteButton');
                
                // Enable delete button only if a specific document is selected
                deleteButton.disabled = !select.value;
            }
            
            async function deleteSelectedDocument() {
                const select = document.getElementById('documentSelect');
                const documentId = select.value;
                
                if (!documentId) {
                    alert('Please select a document to delete.');
                    return;
                }
                
                const selectedOption = select.options[select.selectedIndex];
                const documentName = selectedOption.textContent;
                
                // Confirm deletion
                if (!confirm(`Are you sure you want to delete "${documentName}"?\n\nThis action cannot be undone and will remove all data for this document.`)) {
                    return;
                }
                
                try {
                    const deleteButton = document.getElementById('deleteButton');
                    deleteButton.disabled = true;
                    deleteButton.textContent = '🗑️ Deleting...';
                    
                    const response = await fetch(`/delete-document/?document_id=${encodeURIComponent(documentId)}`, {
                        method: 'DELETE'
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert(`Successfully deleted "${result.document_name}"\n\nRemoved ${result.chunks_deleted} chunks from the database.`);
                        
                        // Reload documents list
                        await loadDocuments();
                        
                        // Clear any existing results if the deleted document was selected
                        document.getElementById('results').innerHTML = '';
                    } else {
                        alert(`Error deleting document: ${result.detail || 'Unknown error'}`);
                    }
                } catch (error) {
                    console.error('Error deleting document:', error);
                    alert(`Error deleting document: ${error.message}`);
                } finally {
                    const deleteButton = document.getElementById('deleteButton');
                    deleteButton.textContent = '🗑️ Delete Selected';
                    updateDeleteButtonState();
                }
            }
            
            async function askQuestion() {
                const query = document.getElementById('queryInput').value.trim();
                const documentId = document.getElementById('documentSelect').value;
                
                if (!query) {
                    alert('Please enter a question');
                    return;
                }
                
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                
                loading.style.display = 'block';
                results.innerHTML = '';
                
                try {
                    const url = new URL('/ask/', window.location.origin);
                    url.searchParams.append('query', query);
                    if (documentId) {
                        url.searchParams.append('document_id', documentId);
                    }
                    
                    const response = await fetch(url, { method: 'POST' });
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResults(data);
                    } else {
                        results.innerHTML = '<div class="answer-container" style="border-left-color: #f44336;"><strong>Error:</strong> ' + data.detail + '</div>';
                    }
                } catch (error) {
                    results.innerHTML = '<div class="answer-container" style="border-left-color: #f44336;"><strong>Error:</strong> ' + error.message + '</div>';
                } finally {
                    loading.style.display = 'none';
                }
            }
            
            function displayResults(data) {
                const results = document.getElementById('results');
                
                let html = '<div class="question"><strong>Question:</strong> ' + data.query + '</div>';
                html += '<div class="answer-container">';
                html += '<h3>🤖 AI Answer:</h3>';
                html += '<p>' + data.answer.replace(/\\n/g, '<br>') + '</p>';
                html += '</div>';
                
                if (data.sources && data.sources.length > 0) {
                    html += '<div class="sources">';
                    html += '<h3>📚 Sources (' + data.total_sources + ' relevant chunks):</h3>';
                    
                    data.sources.forEach((source, index) => {
                        html += '<div class="source-item">';
                        html += '<strong>Source ' + (index + 1) + '</strong> ';
                        html += '<span style="color: #666;">(Relevance: ' + (source.relevance_score * 100).toFixed(1) + '%)</span>';
                        html += '<p style="margin: 0.5rem 0; font-style: italic;">' + source.content_preview + '</p>';
                        if (source.metadata.document_id) {
                            html += '<small style="color: #888;">Document: ' + source.metadata.document_id + '</small>';
                        }
                        html += '</div>';
                    });
                    
                    html += '</div>';
                }
                
                results.innerHTML = html;
            }
            
            // Load documents on page load
            loadDocuments();
            
            // Add event listener to document select to update delete button state
            document.getElementById('documentSelect').addEventListener('change', updateDeleteButtonState);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/documents/")
async def list_documents():
    """Get list of all uploaded documents"""
    if not collection:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    try:
        # Get all documents
        results = collection.get()
        
        # Extract unique document IDs
        document_ids = set()
        document_info = {}
        
        if results['metadatas']:
            for metadata in results['metadatas']:
                if 'document_id' in metadata:
                    doc_id = metadata['document_id']
                    document_name = metadata.get('document_name', 'Unnamed Document')
                    document_ids.add(doc_id)
                    
                    if doc_id not in document_info:
                        document_info[doc_id] = {
                            "document_id": doc_id,
                            "document_name": document_name,
                            "chunk_count": 0
                        }
                    
                    document_info[doc_id]["chunk_count"] += 1
        
        documents_list = list(document_info.values())
        documents_list.sort(key=lambda x: x['document_id'])
        
        return {
            "documents": documents_list,
            "total_documents": len(documents_list),
            "total_chunks": len(results['ids']) if results['ids'] else 0
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.post("/rename-document/")
async def rename_document(document_id: str, new_name: str):
    """
    Rename an existing document
    
    Args:
        document_id: ID of the document to rename
        new_name: New name for the document
        
    Returns:
        Success confirmation
    """
    if not collection:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    if not new_name.strip():
        raise HTTPException(status_code=400, detail="Document name cannot be empty")
    
    try:
        # Get all chunks for this document
        results = collection.get(where={"document_id": document_id})
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update metadata for all chunks
        updated_metadatas = []
        for metadata in results['metadatas']:
            metadata['document_name'] = new_name.strip()
            updated_metadatas.append(metadata)
        
        # Update in ChromaDB
        collection.update(
            ids=results['ids'],
            metadatas=updated_metadatas
        )
        
        return {
            "status": "success",
            "document_id": document_id,
            "new_name": new_name.strip(),
            "chunks_updated": len(results['ids'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rename document: {str(e)}")


@app.delete("/delete-document/")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the vector database
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        Deletion confirmation with count of removed chunks
    """
    if not collection:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    
    try:
        # Get all chunks for this document
        results = collection.get(where={"document_id": document_id})
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunk_count = len(results['ids'])
        document_name = results['metadatas'][0].get('document_name', 'Unknown Document') if results['metadatas'] else 'Unknown Document'
        
        # Delete all chunks for this document
        collection.delete(where={"document_id": document_id})
        
        logger.info(f"Deleted document {document_id} ({document_name}) with {chunk_count} chunks")
        
        return {
            "status": "success",
            "document_id": document_id,
            "document_name": document_name,
            "chunks_deleted": chunk_count,
            "message": f"Successfully deleted '{document_name}' and all its data"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/status/")
async def get_status():
    """Get system status and configuration"""
    return {
        "status": "running",
        "google_api_configured": bool(Config.GOOGLE_API_KEY),
        "api_key_length": len(Config.GOOGLE_API_KEY) if Config.GOOGLE_API_KEY else 0,
        "api_key_preview": Config.GOOGLE_API_KEY[:10] + "..." if Config.GOOGLE_API_KEY and len(Config.GOOGLE_API_KEY) > 10 else "Not set",
        "chromadb_configured": collection is not None,
        "collection_name": Config.COLLECTION_NAME,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "total_documents": collection.count() if collection else 0
    }


@app.get("/health/")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Document Vectorization System...")
    print("📚 Upload PDFs to convert them into vectorized data")
    print("🔍 Use the search endpoint to find similar content")
    print("⚙️  Configure GOOGLE_API_KEY environment variable for full functionality")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
