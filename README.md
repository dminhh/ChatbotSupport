# 🤖 Chatbot Ecommerce Support - RAG System

Chatbot hỗ trợ khách hàng cho trang thương mại điện tử sử dụng công nghệ RAG (Retrieval-Augmented Generation).

## 📋 Tổng quan

Chatbot tự động trả lời các câu hỏi thường gặp (FAQ) về:
- Vận chuyển & Giao hàng
- Thanh toán
- Đổi trả & Hoàn tiền
- Tài khoản & Bảo mật
- Sản phẩm & Dịch vụ
- Khuyến mãi & Ưu đãi

## 🎯 Flow hoạt động

```
User hỏi câu hỏi
    ↓
Vector search tìm top-k FAQs liên quan (FAISS)
    ↓
Build context từ FAQs tìm được
    ↓
GPT-4 Turbo generate câu trả lời tự nhiên
    ↓
Trả về câu trả lời cho user
```

## 🛠 Công nghệ sử dụng

- **Flask** - Web framework cho API
- **OpenAI GPT-4 Turbo** - Generate câu trả lời tự nhiên
- **OpenAI Embeddings** (text-embedding-3-small) - Chuyển text thành vector
- **FAISS** - Vector similarity search
- **Pydantic** - Request validation
- **Python 3.13**

## 📁 Cấu trúc dự án

```
ChatbotSupport/
├── src/
│   ├── app.py                    # Flask API
│   ├── chatbot.py                # Chatbot RAG logic
│   ├── vector_search.py          # FAISS vector search
│   └── product_vector_indexer.py # Product vector indexing
├── data/
│   ├── knowledge_base.json       # 20 câu FAQ
│   ├── faiss_index.bin           # FAISS index (auto-generated)
│   └── metadata.pkl              # Metadata (auto-generated)
├── database/
│   └── create_product_vectors_table.sql  # SQL script tạo bảng
├── .env                          # Environment variables (không push lên git)
├── .env.example                  # Template cho .env
├── .gitignore                    # Git ignore config
├── requirements.txt              # Python dependencies
└── README.md                     # File này
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd ChatbotSupport
```

### 2. Cài đặt dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Cấu hình environment variables

```bash
# Copy file .env.example thành .env
cp .env.example .env

# Sửa file .env và thêm OpenAI API key của bạn
# OPENAI_API_KEY=sk-your-api-key-here
```

**Lưu ý:** Bạn cần có OpenAI API key. Đăng ký tại: https://platform.openai.com/

### 4. Chạy server

```bash
python3 src/app.py
```

Server sẽ chạy tại: `http://localhost:5001`

## ⚙️ Cấu hình

File `.env` chứa các cấu hình:

```bash
# OpenAI API Key (bắt buộc)
OPENAI_API_KEY=sk-your-api-key-here

# Đường dẫn files
KNOWLEDGE_BASE_PATH=data/knowledge_base.json
INDEX_PATH=data/faiss_index.bin

# Cấu hình chatbot
SIMILARITY_THRESHOLD=0.6      # Ngưỡng similarity (0-1)
TOP_K=3                        # Số FAQs retrieve
GPT_MODEL=gpt-4-turbo          # Model GPT sử dụng

# Cấu hình Flask server
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLASK_DEBUG=False


# Cấu hình Embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

**1. Tạo bảng `product_vectors`:**

```bash
mysql -u root -p ecommerce < database/create_product_vectors_table.sql
```

Hoặc chạy SQL thủ công:

```sql
CREATE TABLE IF NOT EXISTS product_vectors (
    id INT PRIMARY KEY AUTO_INCREMENT,
    product_id INT NOT NULL,
    vector JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_product (product_id)
);

CREATE INDEX idx_product_id ON product_vectors(product_id);
CREATE INDEX idx_updated_at ON product_vectors(updated_at);
```

### 📡 API Endpoints

#### 1. **POST `/build-product-index`** - Build index lần đầu

Vector hóa **TẤT CẢ** products và build FAISS index (force rebuild).

```

**Khi nào dùng:**
- Lần đầu tiên setup hệ thống
- Rebuild toàn bộ index khi có thay đổi lớn
- Khi cần reset lại vectors

---

#### 2. **POST `/update-product-index`** - Update index (incremental)

Chỉ vector hóa **products mới** (chưa có trong `product_vectors`) và update index.

```

**Khi nào dùng:**
- Sau khi thêm sản phẩm mới vào database
- Update định kỳ để đồng bộ products mới
- Tiết kiệm cost OpenAI API (chỉ vector hóa products mới)

---

#### 3. **POST `/chat`** - Chat với bot (streaming)

---

#### 4. **POST `/rebuild-index`** - Rebuild chatbot index

Rebuild FAISS index cho knowledge base (FAQs).


## 👤 Author

Hồ Đức Minh
