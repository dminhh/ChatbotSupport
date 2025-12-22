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

    def _generate_response_stream(self, question: str, context: str):
        """
        Sử dụng GPT để generate câu trả lời tự nhiên với streaming.

        Args:
            question: Câu hỏi của user
            context: Context từ FAQs

        Yields:
            Từng chunk của câu trả lời
        """
        system_prompt = """Bạn là trợ lý ảo thông minh của một trang thương mại điện tử Việt Nam.

PHẠM VI HOẠT ĐỘNG:
- CHỈ hỗ trợ các vấn đề liên quan đến mua sắm trực tuyến: đơn hàng, vận chuyển, thanh toán, sản phẩm, đổi trả, khuyến mãi...
- CHỈ trả lời small talk CƠ BẢN: chào hỏi, tạm biệt, cảm ơn
- TUYỆT ĐỐI KHÔNG trả lời về: thể thao, chính trị, giải trí, thời tiết, hay BẤT KỲ chủ đề nào NGOÀI ecommerce

CÁCH XỬ LÝ:
1. Nếu có FAQs liên quan: Dựa vào FAQs để trả lời một cách tự nhiên, thân thiện và chuyên nghiệp
2. Nếu là lời chào/tạm biệt/cảm ơn cơ bản: Chào lại thân thiện, giới thiệu bạn là trợ lý mua sắm, hỏi có thể giúp gì về đơn hàng/sản phẩm
3. Nếu là câu hỏi ngoài phạm vi ecommerce (VD: Ronaldo hay Messi, thời tiết hôm nay...): Lịch sự TỪ CHỐI, nói bạn chỉ hỗ trợ về mua sắm, gợi ý khách hàng hỏi về đơn hàng/sản phẩm
4. Nếu không có FAQs cho câu hỏi về ecommerce: Nói bạn chưa có thông tin này, gợi ý liên hệ hotline 1900-xxxx

QUY TẮC QUAN TRỌNG:
- KHÔNG bịa thông tin không có trong FAQs
- Trả lời ngắn gọn, súc tích, dễ hiểu
- Sử dụng emoji một cách tinh tế nếu phù hợp
- GIỮ ĐÚNG PHẠM VI: Chỉ ecommerce + small talk cơ bản"""

        user_prompt = f"""Câu hỏi của khách hàng: {question}

Các FAQs liên quan:
{context}

Hãy trả lời câu hỏi của khách hàng dựa trên các FAQs trên."""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API: {e}")
            yield "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."

    def chat_stream(self, question: str):
        """
        Xử lý câu hỏi của user và trả về câu trả lời dạng stream.

        Args:
            question: Câu hỏi của user

        Yields:
            Từng chunk của câu trả lời
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

        # Bước 3: Build context từ FAQs
        if relevant_results:
            context = self._build_context(relevant_results)
        else:
            # Không có FAQs relevant, GPT sẽ tự xử lý (small talk hoặc từ chối)
            context = "(Không có FAQs liên quan)"

        # Bước 4 & 5: Stream response từ GPT
        # GPT sẽ tự xử lý cả small talk và từ chối câu hỏi ngoài scope
        for chunk in self._generate_response_stream(question, context):
            yield chunk

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

        print(f"\n💬 Trả lời:")
        for chunk in chatbot.chat_stream(question):
            print(chunk, end='', flush=True)
        print()  # Newline sau khi stream xong

    print(f"\n{'=' * 60}")
    print("✅ Demo hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    main()
