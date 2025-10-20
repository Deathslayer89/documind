#!/usr/bin/env python3
"""
Test script for Gemini API integration.
This script validates that the Gemini API can be used for embeddings and chat.
"""

import os
import sys

def test_gemini_imports():
    """Test that Gemini libraries can be imported."""
    print("🧪 Testing Gemini API Imports...")

    try:
        import google.generativeai as genai
        print("✅ google.generativeai imported")
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        print("Run: pip install google-generativeai")
        return False

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        print("✅ LangChain Google GenAI imported")
    except ImportError as e:
        print(f"❌ Failed to import LangChain Google GenAI: {e}")
        print("Run: pip install langchain-google-genai")
        return False

    return True

def test_gemini_api_key():
    """Test that Gemini API key is set and working."""
    print("\n🧪 Testing Gemini API Key...")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No GOOGLE_API_KEY found in environment")
        print("Set: export GOOGLE_API_KEY='your-key-here'")
        return False

    print(f"✅ API key found (starts with: {api_key[:10]}...)")

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Test basic model access
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini model initialized")

        return True
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")
        return False

def test_embeddings():
    """Test Gemini embeddings functionality."""
    print("\n🧪 Testing Gemini Embeddings...")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        result = embeddings.embed_query(test_text)

        if isinstance(result, list) and len(result) > 0:
            print(f"✅ Embedding generated successfully (dimension: {len(result)})")
            return True
        else:
            print(f"❌ Unexpected embedding result: {result}")
            return False

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def test_chat_model():
    """Test Gemini chat model functionality."""
    print("\n🧪 Testing Gemini Chat Model...")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key
        )

        # Test basic chat
        response = llm.invoke("What is computer science in one sentence?")

        if hasattr(response, 'content') and response.content:
            print(f"✅ Chat model response: {response.content[:100]}...")
            return True
        else:
            print(f"❌ Unexpected chat response: {response}")
            return False

    except Exception as e:
        print(f"❌ Chat model test failed: {e}")
        return False

def main():
    """Run all Gemini tests."""
    print("🚀 Starting Gemini API Integration Tests")
    print("=" * 50)

    # Change to project directory if needed
    if os.path.exists("rag_project"):
        os.chdir("rag_project")

    # Run tests
    tests = [
        ("Gemini Imports", test_gemini_imports),
        ("API Key Setup", test_gemini_api_key),
        ("Embeddings", test_embeddings),
        ("Chat Model", test_chat_model)
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
    print("📊 GEMINI TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Gemini API is ready for use!")
        print("\n📋 Next Steps:")
        print("1. Set GOOGLE_API_KEY in your .env file")
        print("2. Run: pip install -r requirements.txt")
        print("3. Run: streamlit run app.py")
        print("\n✅ Your RAG system with Gemini is ready!")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("\n🔧 To fix issues:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Get API key from: https://makersuite.google.com/app/apikey")
        print("3. Set environment variable: export GOOGLE_API_KEY='your-key'")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)