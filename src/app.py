"""
Flask API cho chatbot ecommerce hỗ trợ khách hàng.

Endpoints:
- POST /chat - Gửi câu hỏi và nhận câu trả lời
- POST /rebuild-index - Rebuild FAISS index khi cập nhật knowledge base
- POST /build-product-index - Build product vector index lần đầu (force rebuild)
- POST /update-product-index - Update product vector index (incremental)
- POST /visual_search - Visual search (tìm sản phẩm bằng ảnh)
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from pydantic import BaseModel, Field, ValidationError
from chatbot import ChatbotRAG
from product_vector_indexer import ProductVectorIndexer
from product_visual_search import ProductVisualSearch
import os
import json
from dotenv import load_dotenv


# Pydantic models cho request validation
class ChatRequest(BaseModel):
    """Request model cho /chat endpoint."""
    question: str = Field(..., min_length=1, description="Câu hỏi của user")
    debug: bool = Field(default=False, description="Debug mode")

class VisualSearchRequest(BaseModel):
    """Request model cho /visual_search endpoint."""
    image_base64: str = Field(..., min_length=1, description="Ảnh dạng base64 string")
    top_k: int = Field(default=10, ge=0, le=10, description="Số lượng kết quả (0-10)")

# Load environment variables
load_dotenv()

# Khởi tạo Flask app
app = Flask(__name__)

# Khởi tạo chatbot (singleton)
chatbot = None

# Khởi tạo product indexer (singleton)
product_indexer = None

# Khởi tạo visual search (singleton)
visual_search = None


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


def get_product_indexer():
    """Khởi tạo product indexer (lazy loading)."""
    global product_indexer
    if product_indexer is None:
        try:
            db_config = {
                'host': os.getenv("DB_HOST", "localhost"),
                'port': int(os.getenv("DB_PORT", "3306")),
                'database': os.getenv("DB_NAME", "ecommerce"),
                'user': os.getenv("DB_USER", "root"),
                'password': os.getenv("DB_PASSWORD", "dm17102002")
            }
            product_indexer = ProductVectorIndexer(
                db_config=db_config,
                embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536"))
            )
            print("✓ Product indexer initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing product indexer: {e}")
            raise
    return product_indexer


def get_visual_search():
    """Khởi tạo visual search (lazy loading)."""
    global visual_search
    if visual_search is None:
        try:
            indexer = get_product_indexer()
            visual_search = ProductVisualSearch(
                product_indexer=indexer,
                vision_model=os.getenv("VISION_MODEL", "gpt-4o"),
                max_results=10,
                score_threshold=float(os.getenv("VISUAL_SEARCH_THRESHOLD", "0.5"))
            )
            print("✓ Visual search initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing visual search: {e}")
            raise
    return visual_search


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat với bot - Streaming response.

    Request body:
    {
        "question": "câu hỏi của user"
    }

    Response:
    Server-Sent Events (SSE) stream - từng chunk text được gửi về
    """
    try:
        # Validate request với Pydantic
        data = request.get_json()
        chat_request = ChatRequest(**data)

        # Get chatbot
        bot = get_chatbot()

        # Stream generator function
        def generate():
            try:
                for chunk in bot.chat_stream(chat_request.question):
                    # Format: Server-Sent Events
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                # Signal end of stream
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

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


@app.route('/build-product-index', methods=['POST'])
def build_product_index():
    """
    Build product index lần đầu (force rebuild toàn bộ).
    Vector hóa tất cả products và lưu vào product_vectors, sau đó build FAISS index.

    Response:
    {
        "success": true,
        "message": "Product index built successfully",
        "stats": {
            "total_products": 100,
            "total_vectors": 100,
            "vectorized_percentage": 100.0
        }
    }
    """
    try:
        indexer = get_product_indexer()
        indexer.update_index(force_rebuild=True)

        stats = indexer.get_stats()

        return jsonify({
            "success": True,
            "message": "Product index built successfully",
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/update-product-index', methods=['POST'])
def update_product_index():
    """
    Update product index (incremental).
    Chỉ vector hóa products mới chưa có trong product_vectors, sau đó rebuild index.

    Response:
    {
        "success": true,
        "message": "Product index updated successfully",
        "stats": {
            "total_products": 100,
            "total_vectors": 100,
            "vectorized_percentage": 100.0
        }
    }
    """
    try:
        indexer = get_product_indexer()
        indexer.update_index(force_rebuild=False)

        stats = indexer.get_stats()

        return jsonify({
            "success": True,
            "message": "Product index updated successfully",
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/visual_search', methods=['POST'])
def visual_search_endpoint():
    """
    Visual search - Tìm kiếm sản phẩm bằng ảnh.

    Request body:
    {
        "image_base64": "base64_string...",
        "top_k": 5  // Optional, default 10, max 10
    }

    Response:
    {
        "success": true,
        "results": [
            {
                "product_id": 123,
                "title": "Áo sơ mi nam công sở",
                "score": 0.95,
                "distance": 0.12
            }
        ],
        "total_results": 5
    }
    """
    try:
        # Validate request
        data = request.get_json()
        search_request = VisualSearchRequest(**data)

        # Get visual search instance
        vs = get_visual_search()

        # Perform search
        result = vs.search(
            image_base64=search_request.image_base64,
            top_k=search_request.top_k
        )

        # Return kết quả
        # 200 OK: Request được xử lý thành công (kể cả khi không tìm thấy sản phẩm)
        # Chỉ trả 400/500 khi có lỗi thực sự (exception)
        return jsonify(result), 200

    except ValidationError as e:
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
    print("  POST /rebuild-index - Rebuild chatbot index")
    print("  POST /build-product-index - Build product index (force rebuild)")
    print("  POST /update-product-index - Update product index (incremental)")
    print("  POST /visual_search - Visual search (tìm sản phẩm bằng ảnh)")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)
