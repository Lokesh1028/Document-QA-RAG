#!/usr/bin/env python3
"""
Setup Checker for Document Q&A System
=====================================

This script helps verify that all dependencies and configurations
are properly set up before running the Streamlit app.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_virtual_environment():
    """Check if virtual environment exists and is activated"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment exists")
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Virtual environment is activated")
            return True
        else:
            print("⚠️  Virtual environment exists but not activated")
            print("   Run: source .venv/bin/activate (macOS/Linux) or .venv\\Scripts\\activate (Windows)")
            return False
    else:
        print("❌ Virtual environment not found")
        print("   Run: python3 -m venv .venv")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to their import names
    packages = {
        'streamlit': 'streamlit',
        'pypdf': 'pypdf', 
        'chromadb': 'chromadb',
        'google-generativeai': 'google.generativeai',
        'langchain-text-splitters': 'langchain_text_splitters',
        'python-dotenv': 'dotenv'
    }
    
    missing_packages = []
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} - Installed")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    return True

def check_env_file():
    """Check .env file and API key"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Create .env file with: GOOGLE_API_KEY=your_api_key_here")
        return False
    
    print("✅ .env file exists")
    
    # Try to load and check API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print(f"✅ GOOGLE_API_KEY found (length: {len(api_key)})")
            
            # Google API keys are typically 39-40 characters long and start with "AIza"
            if len(api_key) >= 35 and api_key.startswith("AIza"):
                print("✅ API key format looks valid")
                return True
            elif api_key == "your_google_api_key_here" or api_key == "AIzaSyDafP9Ja4iEkupT8-LwZeH_EGOV5WpowTo":
                print("❌ API key appears to be a placeholder/example")
                print("   Get your real API key from: https://aistudio.google.com/app/apikey")
                return False
            else:
                print("⚠️  API key format may be incorrect")
                print(f"   Expected: 39-40 characters starting with 'AIza'")
                print(f"   Got: {len(api_key)} characters starting with '{api_key[:4]}...'")
                print("   Verify at: https://aistudio.google.com/app/apikey")
                return False
        else:
            print("❌ GOOGLE_API_KEY not found in .env file")
            return False
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = ['streamlit_app.py', 'requirements.txt']
    all_exist = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def check_vector_db():
    """Check ChromaDB setup"""
    vector_db_path = Path("./vector_db")
    if vector_db_path.exists():
        print("✅ Vector database directory exists")
        files = list(vector_db_path.glob("*"))
        if files:
            print(f"✅ Vector database has {len(files)} files")
        else:
            print("ℹ️  Vector database is empty (normal for first run)")
    else:
        print("ℹ️  Vector database will be created on first run")
    return True

def main():
    """Run all checks"""
    print("🔍 Document Q&A System - Setup Checker")
    print("=" * 45)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Required Files", check_files),
        ("Vector Database", check_vector_db),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        result = check_func()
        all_passed = all_passed and result
    
    print("\n" + "=" * 45)
    if all_passed:
        print("🎉 All checks passed! You're ready to run the app.")
        print("\n🚀 To start the app:")
        print("   ./run_streamlit.sh (macOS/Linux)")
        print("   run_streamlit.bat (Windows)")
        print("   OR: GOOGLE_API_KEY=\"your_key\" streamlit run streamlit_app.py")
    else:
        print("⚠️  Some issues found. Please fix them before running the app.")
        print("\n📚 See README_STREAMLIT.md for detailed instructions.")

if __name__ == "__main__":
    main()
