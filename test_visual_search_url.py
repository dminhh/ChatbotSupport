"""
Test visual search với ảnh từ URL.
"""

import requests
import base64
import json


def download_and_encode_image(url):
    """Download ảnh từ URL và encode base64."""
    print(f"📥 Downloading image from: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Encode to base64
    img_base64 = base64.b64encode(response.content).decode('utf-8')
    print(f"✅ Image downloaded: {len(img_base64)} characters")

    return img_base64


def test_visual_search_with_url(image_url, top_k=5):
    """Test visual search API với ảnh từ URL."""
    print("=" * 60)
    print("🧪 TESTING VISUAL SEARCH API")
    print("=" * 60)

    try:
        # Download và encode ảnh
        image_base64 = download_and_encode_image(image_url)

        # Prepare request
        api_url = "http://localhost:5001/visual_search"
        payload = {
            "image_base64": image_base64,
            "top_k": top_k
        }

        print(f"\n🔍 Calling API: POST {api_url}")
        print(f"Request: top_k={top_k}")

        # Call API
        response = requests.post(api_url, json=payload, timeout=120)

        print(f"\n📊 Response Status: {response.status_code}")

        # Parse response
        result = response.json()

        # Print results
        print("\n" + "=" * 60)
        print("📋 RESULTS")
        print("=" * 60)

        print(f"\nSuccess: {result.get('success')}")

        if result.get('llm_analysis'):
            print("\n🤖 LLM Analysis:")
            analysis = result['llm_analysis']
            print(f"  Product Name: {analysis.get('product_name')}")
            print(f"  Description: {analysis.get('description')}")
            if analysis.get('keywords'):
                print(f"  Keywords: {', '.join(analysis.get('keywords', []))}")

        if result.get('search_query'):
            print(f"\n🔍 Search Query:")
            print(f"  {result['search_query']}")

        if result.get('results'):
            print(f"\n🎯 Found {result['total_results']} products:")
            for i, product in enumerate(result['results'], 1):
                print(f"\n  {i}. [ID: {product['product_id']}] {product['title']}")
                print(f"     Score: {product['score']:.4f} | Distance: {product['distance']:.4f}")

        if not result.get('success'):
            print(f"\n❌ Error: {result.get('error')}")

        print("\n" + "=" * 60)
        print("✅ Test completed!")
        print("=" * 60)

        # Save full response
        filename = 'visual_search_test_result.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full response saved to: {filename}")

        return result

    except requests.exceptions.Timeout:
        print("\n❌ Request timeout! API took too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection error! Is the server running?")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test với ảnh áo thun từ một URL công khai
    # Bạn có thể thay đổi URL này bằng ảnh sản phẩm khác

    test_urls = [
        # Ảnh áo thun đơn giản
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400",

        # Hoặc dùng URL ảnh khác
        # "https://your-image-url.jpg"
    ]

    for url in test_urls[:1]:  # Test URL đầu tiên
        print(f"\n\n{'='*60}")
        print(f"Testing with image: {url}")
        print(f"{'='*60}\n")

        result = test_visual_search_with_url(url, top_k=5)

        if result and result.get('success'):
            print("\n✅ Test PASSED!")
        else:
            print("\n⚠️ Test có vấn đề - kiểm tra log phía trên")
