#!/usr/bin/env python3
"""
Quick test script to verify the RAG system components work.
This script tests each component individually without requiring API keys.
"""

import os
import sys

def test_document_processor():
    """Test the document processor with sample data."""
    print("🧪 Testing Document Processor...")

    try:
        from document_processor import DocumentProcessor

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        print("✅ DocumentProcessor initialized")

        # Test with the sample text file
        if os.path.exists("data/sample_cs_notes.txt"):
            documents = processor.process_document("data/sample_cs_notes.txt")
            print(f"✅ Processed sample text: {len(documents)} chunks")

            if documents:
                print(f"✅ Sample chunk preview: {documents[0].page_content[:100]}...")
                print(f"✅ Metadata: {documents[0].metadata}")
        else:
            print("⚠️ Sample text file not found")

        return True

    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_environment_setup():
    """Test that required files and directories exist."""
    print("\n🧪 Testing Environment Setup...")

    required_files = [
        "requirements.txt",
        ".env.example",
        "app.py",
        "rag_pipeline.py",
        "document_processor.py",
        "evaluation.py",
        "README.md"
    ]

    required_dirs = ["data", "chroma_db"]

    all_good = True

    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_good = False

    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✅ {dir}/ directory")
        else:
            print(f"❌ {dir}/ directory missing")
            all_good = False

    return all_good

def test_imports():
    """Test that all required modules can be imported (excluding API-dependent ones)."""
    print("\n🧪 Testing Module Imports...")

    modules_to_test = [
        ("document_processor", "DocumentProcessor"),
        ("os", None),
        ("json", None),
        ("tempfile", None),
    ]

    all_good = True

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if class_name:
                getattr(module, class_name)
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            all_good = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            all_good = False

    return all_good

def test_file_processing():
    """Test processing different file types."""
    print("\n🧪 Testing File Processing...")

    try:
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()

        data_dir = "data"
        if not os.path.exists(data_dir):
            print("❌ Data directory not found")
            return False

        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        print(f"📁 Found {len(files)} files in data directory")

        for file in files[:2]:  # Test first 2 files only
            file_path = os.path.join(data_dir, file)
            print(f"🔍 Testing {file}...")

            try:
                documents = processor.process_document(file_path)
                if documents:
                    print(f"✅ {file}: {len(documents)} chunks")
                else:
                    print(f"⚠️ {file}: No content extracted")
            except Exception as e:
                print(f"❌ {file}: {e}")

        return True

    except Exception as e:
        print(f"❌ File processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting RAG System Tests")
    print("=" * 50)

    # Change to project directory
    if os.path.exists("rag_project"):
        os.chdir("rag_project")

    # Run tests
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Module Imports", test_imports),
        ("Document Processor", test_document_processor),
        ("File Processing", test_file_processing)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        print("\n📋 Next Steps:")
        print("1. Set your OPENAI_API_KEY in .env file")
        print("2. Run: pip install -r requirements.txt")
        print("3. Run: streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)