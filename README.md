# Document Vectorization System

A comprehensive Python application that converts raw PDF documents into vectorized data for AI-powered document retrieval and search.

## 🚀 Features

- **PDF Upload & Processing**: Web interface for uploading PDF documents
- **Text Extraction**: Robust text extraction from PDFs using PyMuPDF
- **Intelligent Chunking**: Smart text segmentation using LangChain's text splitters
- **Vector Embeddings**: Generate embeddings using Google's text-embedding-004 model
- **Vector Storage**: Persistent storage using ChromaDB
- **Semantic Search**: Find similar content using vector similarity search
- **RESTful API**: Complete API for document processing and search
- **Web Interface**: Simple HTML interface for easy document uploads

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI API key (for embedding generation)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd dfr
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp config.example .env
   # Edit .env file and add your Google AI API key
   ```

4. **Get Google AI API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

## 🚦 Quick Start

1. **Start the application**:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:8000`

3. **Upload a PDF**:
   - Use the web interface to upload a PDF document
   - The system will automatically process and vectorize the document

4. **Search documents**:
   - Use the `/search/` endpoint to find similar content
   - Example: `http://localhost:8000/search/?query=your search query&limit=5`

## 📖 API Documentation

### Endpoints

#### `GET /`
- **Description**: Web interface for document uploads
- **Response**: HTML upload page

#### `POST /upload-pdf/`
- **Description**: Upload and process a PDF document
- **Parameters**: 
  - `file`: PDF file (multipart/form-data)
- **Response**: Processing result with document ID and statistics

#### `GET /search/`
- **Description**: Search for similar document content
- **Parameters**:
  - `query`: Search query text
  - `limit`: Maximum number of results (default: 5)
- **Response**: List of similar document chunks with metadata

#### `GET /status/`
- **Description**: Get system status and configuration
- **Response**: System configuration and database statistics

#### `GET /health/`
- **Description**: Health check endpoint
- **Response**: System health status

### Example Usage

**Upload a PDF**:
```bash
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

**Search documents**:
```bash
curl -X GET "http://localhost:8000/search/?query=machine learning&limit=3"
```

## 🏗️ Architecture

The system follows a pipeline architecture:

```
PDF Upload → Text Extraction → Text Chunking → Embedding Generation → Vector Storage
     ↓              ↓              ↓                    ↓                ↓
  FastAPI      PyMuPDF      LangChain        Google AI API      ChromaDB
```

### Components

1. **Web Layer**: FastAPI application with upload endpoints
2. **Text Processing**: PyMuPDF for PDF text extraction
3. **Chunking**: LangChain's RecursiveCharacterTextSplitter
4. **Embedding**: Google's text-embedding-004 model
5. **Storage**: ChromaDB for vector storage and retrieval

## ⚙️ Configuration

Environment variables (configure in `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key (required) | None |
| `CHROMADB_PATH` | Path to ChromaDB storage | `./vector_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `pdf_documents` |
| `CHUNK_SIZE` | Text chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `100` |

## 🔍 Processing Pipeline

When you upload a PDF, the system:

1. **Validates** the file type and saves it temporarily
2. **Extracts** text from all pages using PyMuPDF
3. **Chunks** the text into manageable pieces with overlap
4. **Generates** vector embeddings for each chunk using Google AI
5. **Stores** the vectors and metadata in ChromaDB
6. **Returns** processing statistics and document ID

## 📊 Vector Search

The search functionality:

1. **Converts** your query into a vector embedding
2. **Searches** the vector database for similar content
3. **Returns** the most relevant document chunks
4. **Includes** metadata and similarity scores

## 🛠️ Development

**Run in development mode**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Check system status**:
```bash
curl http://localhost:8000/status/
```

## 📝 Supported File Types

Currently supports:
- PDF documents (.pdf)

## 🔧 Troubleshooting

**Common Issues**:

1. **Missing Google API Key**:
   - Error: "Google API key not configured"
   - Solution: Set `GOOGLE_API_KEY` in your `.env` file

2. **ChromaDB Initialization Failed**:
   - Check write permissions for the database directory
   - Ensure sufficient disk space

3. **PDF Processing Errors**:
   - Verify the PDF is not corrupted
   - Check if the PDF is password-protected

4. **Out of Memory**:
   - Reduce `CHUNK_SIZE` for large documents
   - Process smaller files or increase system memory

## 📚 Dependencies

- **FastAPI**: Modern web framework for building APIs
- **PyMuPDF**: PDF text extraction
- **LangChain**: Text processing and chunking
- **ChromaDB**: Vector database for storage and retrieval
- **Google Generative AI**: Embedding generation
- **python-dotenv**: Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for more details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on the repository

---

**Built with ❤️ using Python, FastAPI, and modern AI technologies**
