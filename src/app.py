"""
Flask API cho chatbot ecommerce hỗ trợ khách hàng.

Endpoints:
- POST /chat - Gửi câu hỏi và nhận câu trả lời
- POST /rebuild-index - Rebuild FAISS index khi cập nhật knowledge base
"""

from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from chatbot import ChatbotRAG
import os
from dotenv import load_dotenv


# Pydantic models cho request validation
class ChatRequest(BaseModel):
    """Request model cho /chat endpoint."""
    question: str = Field(..., min_length=1, description="Câu hỏi của user")
    debug: bool = Field(default=False, description="Debug mode")

# Load environment variables
load_dotenv()

# Khởi tạo Flask app
app = Flask(__name__)

# Khởi tạo chatbot (singleton)
chatbot = None


def get_chatbot():
    """Lazy initialization của chatbot."""
    global chatbot
    if chatbot is None:
        try:
            chatbot = ChatbotRAG(
                knowledge_base_path=os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.json"),
                index_path=os.getenv("INDEX_PATH", "data/faiss_index.bin"),
                similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
                top_k=int(os.getenv("TOP_K", "3")),
                model=os.getenv("GPT_MODEL", "gpt-3.5-turbo")
            )
            print("✓ Chatbot initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            raise
    return chatbot


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat với bot.

    Request body:
    {
        "question": "câu hỏi của user",
        "debug": false  // optional, default false
    }

    Response:
    {
        "success": true,
        "data": {
            "question": "câu hỏi của user",
            "answer": "câu trả lời của bot",
            "is_confident": true/false,
            "search_results": [...]  // nếu debug=true
        }
    }
    """
    try:
        # Validate request với Pydantic
        data = request.get_json()
        chat_request = ChatRequest(**data)

        # Get response from chatbot
        bot = get_chatbot()
        result = bot.chat(chat_request.question, debug=chat_request.debug)

        return jsonify({
            "success": True,
            "data": result
        }), 200

    except ValidationError as e:
        # Pydantic validation error
        return jsonify({
            "success": False,
            "error": "Invalid request",
            "details": e.errors()
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/rebuild-index', methods=['POST'])
def rebuild_index():
    """
    Rebuild FAISS index khi knowledge base được cập nhật.

    Response:
    {
        "success": true,
        "message": "Index rebuilt successfully"
    }
    """
    try:
        bot = get_chatbot()
        bot.rebuild_index()

        return jsonify({
            "success": True,
            "message": "Index rebuilt successfully"
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # Lấy config từ environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    print("=" * 60)
    print("🤖 CHATBOT ECOMMERCE API")
    print("=" * 60)
    print(f"Server: http://{host}:{port}")
    print("\nEndpoints:")
    print("  POST /chat - Chat với bot")
    print("  POST /rebuild-index - Rebuild index")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)
