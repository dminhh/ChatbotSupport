"""
Chatbot hỗ trợ FAQ cho ecommerce sử dụng RAG (Retrieval-Augmented Generation).

Flow:
1. User hỏi câu hỏi
2. Vector search tìm top-k FAQs liên quan
3. Build context từ FAQs
4. Gửi context + question vào GPT
5. GPT generate câu trả lời tự nhiên
"""

import os
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from vector_search import VectorSearch

# Load environment variables
load_dotenv()


class ChatbotRAG:
    """Chatbot sử dụng RAG để trả lời câu hỏi về ecommerce."""

    def __init__(
        self,
        knowledge_base_path: str = "data/knowledge_base.json",
        index_path: str = "data/faiss_index.bin",
        similarity_threshold: float = 0.6,
        top_k: int = 3,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Khởi tạo chatbot.

        Args:
            knowledge_base_path: Đường dẫn đến file knowledge base JSON
            index_path: Đường dẫn đến FAISS index
            similarity_threshold: Ngưỡng similarity để xác định câu hỏi có relevant không (0-1)
            top_k: Số lượng FAQs liên quan nhất để retrieve
            model: Model OpenAI GPT để sử dụng
        """
        self.vector_search = VectorSearch(knowledge_base_path, index_path)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.model = model

        # Khởi tạo OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY không tìm thấy trong environment variables")

        self.client = OpenAI(api_key=api_key)

        # Template cho fallback response
        self.fallback_response = """Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn.

Bạn có thể hỏi về:
• Chính sách đổi trả
• Thời gian vận chuyển
• Phương thức thanh toán
• Hướng dẫn chọn size

Hoặc liên hệ:
📞 Hotline: 1900-xxxx"""

    def _build_context(self, search_results: List[Dict]) -> str:
        """
        Build context từ kết quả search.

        Args:
            search_results: List các FAQs từ vector search

        Returns:
            Context string để gửi cho GPT
        """
        if not search_results:
            return ""

        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"FAQ {i} (Độ liên quan: {result['similarity']:.2f}):\n"
                f"Câu hỏi: {result['question']}\n"
                f"Trả lời: {result['answer']}\n"
                f"Danh mục: {result['category']}"
            )

        return "\n\n".join(context_parts)

    def _generate_response(self, question: str, context: str) -> str:
        """
        Sử dụng GPT để generate câu trả lời tự nhiên.

        Args:
            question: Câu hỏi của user
            context: Context từ FAQs

        Returns:
            Câu trả lời được generate bởi GPT
        """
        system_prompt = """Bạn là trợ lý ảo thông minh của một trang thương mại điện tử Việt Nam.

Nhiệm vụ của bạn:
1. Dựa vào các FAQs được cung cấp, trả lời câu hỏi của khách hàng một cách tự nhiên, thân thiện và chuyên nghiệp
2. Tổng hợp thông tin từ nhiều FAQs nếu cần
3. Trả lời bằng tiếng Việt
4. Giữ giọng điệu lịch sự, nhiệt tình
5. Nếu FAQs không chứa thông tin để trả lời, hãy thành thật nói rằng bạn chưa có thông tin này

Lưu ý:
- KHÔNG bịa thông tin không có trong FAQs
- Trả lời ngắn gọn, súc tích, dễ hiểu
- Sử dụng emoji một cách tinh tế nếu phù hợp"""

        user_prompt = f"""Câu hỏi của khách hàng: {question}

Các FAQs liên quan:
{context}

Hãy trả lời câu hỏi của khách hàng dựa trên các FAQs trên."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API: {e}")
            return "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."

    def chat(self, question: str, debug: bool = False) -> Dict:
        """
        Xử lý câu hỏi của user và trả về câu trả lời.

        Args:
            question: Câu hỏi của user
            debug: Nếu True, trả về thêm thông tin debug

        Returns:
            Dictionary chứa response và metadata
        """
        # Bước 1 & 2: Vector search tìm top-k FAQs liên quan
        search_results = self.vector_search.search(
            query=question,
            top_k=self.top_k
        )

        # Kiểm tra xem có FAQ nào đủ relevant không
        relevant_results = [
            r for r in search_results
            if r['similarity'] >= self.similarity_threshold
        ]

        response_data = {
            "question": question,
            "answer": "",
            "is_confident": False,
            "search_results": search_results if debug else None
        }

        # Nếu không có FAQ nào đủ relevant, trả về fallback response
        if not relevant_results:
            response_data["answer"] = self.fallback_response
            response_data["is_confident"] = False
            return response_data

        # Bước 3: Build context từ FAQs
        context = self._build_context(relevant_results)

        # Bước 4 & 5: Gửi context + question vào GPT để generate response
        answer = self._generate_response(question, context)

        response_data["answer"] = answer
        response_data["is_confident"] = True

        return response_data

    def rebuild_index(self):
        """Rebuild FAISS index khi knowledge base được cập nhật."""
        self.vector_search.rebuild_index()
        print("✓ Index đã được rebuild thành công!")

    def get_stats(self) -> Dict:
        """Lấy thống kê về vector search index."""
        return self.vector_search.get_stats()


def main():
    """Test chatbot với một số câu hỏi mẫu."""
    print("=" * 60)
    print("🤖 CHATBOT ECOMMERCE - RAG DEMO")
    print("=" * 60)

    # Khởi tạo chatbot
    try:
        chatbot = ChatbotRAG(
            knowledge_base_path="data/knowledge_base.json",
            index_path="data/faiss_index.bin",
            similarity_threshold=0.6,
            top_k=3
        )

        # In thống kê
        stats = chatbot.get_stats()
        print(f"\n📊 Thống kê: {stats['num_questions']} câu hỏi trong knowledge base")
        print(f"🔧 Model: {chatbot.model}")
        print(f"🎯 Similarity threshold: {chatbot.similarity_threshold}")

    except Exception as e:
        print(f"❌ Lỗi khởi tạo chatbot: {e}")
        return

    # Danh sách câu hỏi test
    test_questions = [
        "Tôi muốn biết về chính sách giao hàng",
        "Làm sao để thanh toán khi mua hàng?",
        "Sản phẩm bị lỗi thì đổi như thế nào?",
        "Làm thế nào để mua xe máy?"  # Câu hỏi không liên quan để test fallback
    ]

    # Test từng câu hỏi
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"❓ Câu hỏi {i}: {question}")
        print("-" * 60)

        result = chatbot.chat(question, debug=True)

        print(f"\n💬 Trả lời:")
        print(result['answer'])

        if result['search_results']:
            print(f"\n🔍 Debug - Top FAQs tìm được:")
            for j, res in enumerate(result['search_results'][:3], 1):
                print(f"  {j}. {res['question'][:50]}... "
                      f"(similarity: {res['similarity']:.3f})")

        print(f"\n✓ Confidence: {'Cao' if result['is_confident'] else 'Thấp (fallback)'}")

    print(f"\n{'=' * 60}")
    print("✅ Demo hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    main()
