"""
Test script for Cloudflare Worker LLM API
Tests both direct API calls and via LightRAG server
"""

import requests
import json

# === Configuration ===
CLOUDFLARE_URL = "https://kltn-lightra.vuduylinh150804.workers.dev"
LIGHTRAG_URL = "http://localhost:9621"
API_KEY = "12345678"


def test_cloudflare_original_format():
    """Test Cloudflare Worker with original API format"""
    print("=" * 50)
    print("Test 1: Cloudflare Worker - Original Format")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": "Xin chào, bạn là ai?",
        "systemPrompt": "Bạn là một trợ lý AI thân thiện.",
        "history": []
    }
    
    try:
        response = requests.post(f"{CLOUDFLARE_URL}/", headers=headers, json=data, timeout=60)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
    except Exception as e:
        print(f"Error: {e}")


def test_cloudflare_openai_format():
    """Test Cloudflare Worker with OpenAI-compatible format (non-streaming)"""
    print("\n" + "=" * 50)
    print("Test 2: Cloudflare Worker - OpenAI Format (Non-Streaming)")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3-8b",
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý AI thân thiện."},
            {"role": "user", "content": "Xin chào, bạn là ai?"}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(f"{CLOUDFLARE_URL}/v1/chat/completions", headers=headers, json=data, timeout=60)
        print(f"Status: {response.status_code}")
        result = response.json()
        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content[:500]}...")
        else:
            print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")


def test_cloudflare_streaming():
    """Test Cloudflare Worker with OpenAI-compatible streaming format"""
    print("\n" + "=" * 50)
    print("Test 3: Cloudflare Worker - OpenAI Format (Streaming)")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3-8b",
        "messages": [
            {"role": "user", "content": "Nói 'Xin chào' bằng tiếng Việt"}
        ],
        "stream": True
    }
    
    try:
        response = requests.post(f"{CLOUDFLARE_URL}/v1/chat/completions", headers=headers, json=data, stream=True, timeout=60)
        print(f"Status: {response.status_code}")
        print("Streaming response:")
        
        full_content = ""
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print(f"\n\nFull response length: {len(full_content)} chars")
    except Exception as e:
        print(f"Error: {e}")


def test_lightrag_query():
    """Test LightRAG server with Cloudflare Worker backend"""
    print("\n" + "=" * 50)
    print("Test 4: LightRAG Server - Query API")
    print("=" * 50)
    
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "Xin chào",
        "mode": "naive"
    }
    
    try:
        response = requests.post(f"{LIGHTRAG_URL}/query", headers=headers, json=data, timeout=120)
        print(f"Status: {response.status_code}")
        result = response.json()
        if "response" in result:
            print(f"Response: {result['response'][:500]}...")
        else:
            print(f"Response: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("🚀 Testing Cloudflare Worker LLM Integration\n")
    
    # Test direct Cloudflare Worker API
    test_cloudflare_original_format()
    # test_cloudflare_openai_format()
    # test_cloudflare_streaming()
    
    # # Test via LightRAG (requires server running)
    # print("\n" + "=" * 50)
    # print("Note: Test 4 requires LightRAG server running on port 9621")
    # print("=" * 50)
    # test_lightrag_query()
    
    # print("\n✅ All tests completed!")
