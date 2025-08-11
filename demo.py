#!/usr/bin/env python3
"""
Demo script for the Document Vectorization System

This script demonstrates how to use the document processing pipeline
programmatically without the web interface.

Usage:
    python demo.py path/to/document.pdf
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import DocumentProcessor, Config
import uuid


def demo_process_pdf(pdf_path: str):
    """
    Demonstrate PDF processing pipeline
    
    Args:
        pdf_path: Path to PDF file to process
    """
    
    print("🚀 Document Vectorization Demo")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return
    
    # Check configuration
    print("🔧 Checking configuration...")
    if not Config.GOOGLE_API_KEY:
        print("⚠️  Warning: GOOGLE_API_KEY not set. Embedding generation will fail.")
        print("   Set your API key in a .env file or environment variable.")
    else:
        print("✅ Google API key configured")
    
    print(f"📊 Chunk size: {Config.CHUNK_SIZE}")
    print(f"📊 Chunk overlap: {Config.CHUNK_OVERLAP}")
    print(f"📂 Vector DB path: {Config.CHROMADB_PATH}")
    print()
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    print(f"📄 Processing document: {pdf_path}")
    print(f"🔢 Document ID: {document_id}")
    print()
    
    try:
        # Step 1: Extract text
        print("📝 Step 1: Extracting text from PDF...")
        text = DocumentProcessor.extract_text_from_pdf(pdf_path)
        print(f"   ✅ Extracted {len(text):,} characters")
        
        # Step 2: Chunk text
        print("✂️  Step 2: Chunking text...")
        chunks = DocumentProcessor.chunk_text(text)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Show sample chunks
        print("   📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk[:100].replace('\n', ' ')
            print(f"      Chunk {i+1}: {preview}...")
        
        if Config.GOOGLE_API_KEY:
            # Step 3: Generate embeddings
            print("🧠 Step 3: Generating embeddings...")
            embeddings = DocumentProcessor.generate_embeddings(chunks)
            print(f"   ✅ Generated {len(embeddings)} embeddings")
            print(f"   📊 Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
            
            # Step 4: Store vectors
            print("💾 Step 4: Storing vectors in database...")
            storage_result = DocumentProcessor.store_vectors(chunks, embeddings, document_id)
            print(f"   ✅ Stored {storage_result['chunks_stored']} vectors")
            
            print()
            print("🎉 Processing completed successfully!")
            print(f"📊 Summary:")
            print(f"   • Document ID: {document_id}")
            print(f"   • Text length: {len(text):,} characters")
            print(f"   • Chunks created: {len(chunks)}")
            print(f"   • Embeddings generated: {len(embeddings)}")
            print(f"   • Vectors stored: {storage_result['chunks_stored']}")
            
        else:
            print("⏭️  Skipping embedding generation and storage (API key not configured)")
            print()
            print("✅ Text extraction and chunking completed!")
            print(f"📊 Summary:")
            print(f"   • Text length: {len(text):,} characters")
            print(f"   • Chunks created: {len(chunks)}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return
    
    print()
    print("🔍 To search your documents, start the web server:")
    print("   python main.py")
    print("   Then visit: http://localhost:8000/search/?query=your_search_term")


def main():
    """Main demo function"""
    
    if len(sys.argv) != 2:
        print("Usage: python demo.py <path_to_pdf>")
        print()
        print("Example:")
        print("   python demo.py document.pdf")
        print("   python demo.py /path/to/my/document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    demo_process_pdf(pdf_path)


if __name__ == "__main__":
    main()
