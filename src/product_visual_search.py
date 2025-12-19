"""
Product Visual Search - Tìm kiếm sản phẩm bằng ảnh.

Flow:
1. Nhận ảnh (base64) → Phân tích bằng LLM Vision
2. LLM output: product_name, keywords, description
3. Tạo search query từ output
4. Embedding + FAISS search
5. Return top K products
"""

import base64
import json
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI


class ProductVisualSearch:
    """
    Visual search sử dụng LLM Vision + FAISS index.
    """

    def __init__(
        self,
        product_indexer,
        vision_model: str = "gpt-4o",
        max_results: int = 10,
        score_threshold: float = 0.5
    ):
        """
        Khởi tạo ProductVisualSearch.

        Args:
            product_indexer: Instance của ProductVectorIndexer (đã có FAISS index)
            vision_model: Model OpenAI Vision để phân tích ảnh
            max_results: Số lượng kết quả tối đa trả về
            score_threshold: Ngưỡng điểm tối thiểu (0-1). Chỉ trả về sản phẩm có score >= threshold
        """
        self.product_indexer = product_indexer
        self.vision_model = vision_model
        self.max_results = max_results
        self.score_threshold = score_threshold

        # Khởi tạo OpenAI client
        self.client = OpenAI()

    def analyze_image(self, image_base64: str) -> Optional[Dict]:
        """
        Phân tích ảnh bằng LLM Vision.

        Args:
            image_base64: Ảnh dạng base64 string

        Returns:
            Dictionary chứa product_name, keywords, description
            {
                "product_name": "...",
                "keywords": [...],
                "description": "..."
            }
        """
        system_prompt = """Bạn là AI chuyên phân tích ảnh sản phẩm thời trang và đồ dùng.

Nhiệm vụ: Phân tích ảnh và trả về thông tin sản phẩm dạng JSON.

Output format (JSON):
{
  "product_name": "Tên sản phẩm ngắn gọn",
  "keywords": ["keyword1", "keyword2", ...],
  "description": "Mô tả chi tiết sản phẩm"
}

Yêu cầu:
- product_name: Tên sản phẩm chính xác, ngắn gọn (VD: "Áo sơ mi tay dài", "Quần jean nam")
- keywords: 5-10 từ khóa liên quan (tiếng Việt), bao gồm:
  + Loại sản phẩm
  + Phong cách (casual, formal, streetwear, etc.)
  + Màu sắc
  + Chất liệu (nếu nhìn thấy)
  + Đặc điểm nổi bật
  + Từ khóa tìm kiếm phổ biến
- description: Mô tả chi tiết về sản phẩm (màu sắc, kiểu dáng, phong cách, dịp sử dụng)

Lưu ý:
- Chỉ trả về JSON, không giải thích thêm
- Nếu không phải sản phẩm thời trang/đồ dùng, trả về null"""

        try:
            # Đảm bảo base64 string đúng format
            if not image_base64.startswith('data:image'):
                # Thêm prefix nếu chưa có
                image_base64 = f"data:image/jpeg;base64,{image_base64}"

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Phân tích ảnh sản phẩm này và trả về JSON như yêu cầu."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()

            # Xử lý trường hợp LLM trả về markdown code block
            if content.startswith('```'):
                # Remove markdown code block
                content = content.replace('```json', '').replace('```', '').strip()

            result = json.loads(content)

            # Validate output
            if result and 'product_name' in result and 'keywords' in result and 'description' in result:
                return result
            else:
                print(f"⚠️ LLM output thiếu fields!")
                print(f"Raw content: {content}")
                print(f"Parsed result: {result}")
                print(f"Missing fields: {set(['product_name', 'keywords', 'description']) - set(result.keys() if result else [])}")
                return None

        except json.JSONDecodeError as e:
            print(f"❌ Lỗi parse JSON từ LLM: {e}")
            print(f"Raw content from LLM:\n{content}")
            return None
        except Exception as e:
            print(f"❌ Lỗi khi phân tích ảnh: {e}")
            return None

    def create_search_query(self, llm_output: Dict) -> str:
        """
        Tạo search query từ LLM output.

        Args:
            llm_output: Dictionary từ analyze_image()

        Returns:
            Search query string
        """
        product_name = llm_output.get('product_name', '')
        description = llm_output.get('description', '')
        keywords = llm_output.get('keywords', [])

        # Kết hợp: product_name + description + top keywords
        parts = [product_name, description]

        # Thêm top 5 keywords
        if keywords:
            parts.append(' '.join(keywords[:5]))

        query = ' '.join(parts)
        return query

    def search(self, image_base64: str, top_k: int = None) -> Dict:
        """
        Visual search - tìm sản phẩm từ ảnh.

        Args:
            image_base64: Ảnh dạng base64 string
            top_k: Số lượng kết quả (None = dùng max_results)

        Returns:
            {
                "success": True/False,
                "llm_analysis": {...},
                "search_query": "...",
                "results": [
                    {
                        "product_id": 123,
                        "title": "...",
                        "score": 0.95,
                        "distance": 0.12
                    }
                ],
                "total_results": 5
            }
        """
        if top_k is None:
            top_k = self.max_results

        # Giới hạn top_k
        top_k = min(max(top_k, 0), self.max_results)

        try:
            # Bước 1: Phân tích ảnh bằng LLM
            print("🔍 Analyzing image with LLM...")
            llm_output = self.analyze_image(image_base64)

            if not llm_output:
                return {
                    "success": False,
                    "error": "Ảnh này không phải sản phẩm thời trang hoặc đồ dùng. Vui lòng thử với ảnh sản phẩm khác (áo, quần, giày, phụ kiện...).",
                    "results": [],
                    "total_results": 0
                }

            print(f"✅ LLM analysis: {llm_output['product_name']}")

            # Bước 2: Tạo search query
            search_query = self.create_search_query(llm_output)
            print(f"🔍 Search query: {search_query}")

            # Bước 3: Embedding query
            print("🔄 Creating embedding...")
            query_embedding = self.product_indexer._create_embeddings_batch([search_query])[0]

            # Bước 4: FAISS search
            if self.product_indexer.index is None or self.product_indexer.index.ntotal == 0:
                return {
                    "success": False,
                    "error": "Product index chưa được build. Vui lòng gọi /build-product-index trước.",
                    "results": [],
                    "total_results": 0
                }

            print(f"🔍 Searching in FAISS index ({self.product_indexer.index.ntotal} products)...")

            # Search
            query_vector = query_embedding.reshape(1, -1)
            distances, indices = self.product_indexer.index.search(query_vector, top_k)

            # Bước 5: Build results với threshold filtering
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS trả về -1 nếu không đủ kết quả
                    continue

                # Convert distance sang similarity score (0-1, càng cao càng giống)
                score = 1.0 / (1.0 + float(dist))

                # Filter theo threshold
                if score < self.score_threshold:
                    continue

                results.append({
                    "product_id": int(self.product_indexer.product_ids[idx]),
                    "title": self.product_indexer.titles[idx],
                    "score": round(score, 4),
                    "distance": round(float(dist), 4)
                })

            print(f"✅ Found {len(results)} products (threshold: {self.score_threshold})")

            return {
                "success": True,
                "results": results,
                "total_results": len(results)
            }

        except Exception as e:
            print(f"❌ Lỗi trong visual search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_results": 0
            }
