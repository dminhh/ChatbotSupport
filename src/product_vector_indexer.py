"""
Product Vector Indexer - Vector hóa products và build FAISS index.
Lưu vectors vào bảng product_vectors trong MySQL.

Flow mới:
1. Lấy ảnh nền (ANH_NEN) của products từ bảng images
2. Phân tích ảnh bằng LLM Vision → JSON {product_name, keywords, description}
3. Tạo text mô tả từ JSON
4. Embedding text mô tả và lưu vào product_vectors
"""

import json
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss
from openai import OpenAI
import mysql.connector
from mysql.connector import Error


class ProductVectorIndexer:
    """
    Quản lý việc vector hóa products và build FAISS index.

    Flow:
    1. Lấy products từ bảng products (id, title)
    2. Vector hóa title bằng OpenAI Embeddings
    3. Lưu vectors vào bảng product_vectors (JSON format)
    4. Build FAISS index từ product_vectors
    """

    def __init__(
        self,
        db_config: Dict[str, str],
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536,
        index_path: str = "data/product_index.bin",
        metadata_path: str = "data/product_metadata.pkl"
    ):
        """
        Khởi tạo ProductVectorIndexer.

        Args:
            db_config: Dictionary chứa thông tin kết nối database
                {
                    'host': 'localhost',
                    'port': 3306,
                    'database': 'ecommerce',
                    'user': 'root',
                    'password': 'password'
                }
            embedding_model: Model OpenAI Embeddings
            dimension: Số chiều của vector (1536 cho text-embedding-3-small)
            index_path: Đường dẫn lưu FAISS index
            metadata_path: Đường dẫn lưu metadata (product_ids, titles)
        """
        self.db_config = db_config
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Khởi tạo OpenAI client
        self.client = OpenAI()

        # FAISS index và metadata
        self.index: Optional[faiss.Index] = None
        self.product_ids: List[int] = []
        self.titles: List[str] = []

        # Load index nếu có
        self._load_index_if_exists()

    def _get_db_connection(self):
        """Tạo kết nối đến MySQL database."""
        try:
            connection = mysql.connector.connect(
                host=self.db_config['host'],
                port=self.db_config.get('port', 3306),
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )

            if connection.is_connected():
                return connection
        except Error as e:
            print(f"❌ Lỗi kết nối MySQL: {e}")
            raise

    def _analyze_image_from_url(self, image_url: str) -> Optional[Dict]:
        """
        Phân tích ảnh sản phẩm bằng LLM Vision từ URL.
        Sử dụng CÙNG format như ProductVisualSearch để đảm bảo consistency.

        Args:
            image_url: URL công khai của ảnh sản phẩm

        Returns:
            Dictionary chứa product_name, keywords, description
            {
                "product_name": "...",
                "keywords": [...],
                "description": "..."
            }
            Hoặc None nếu có lỗi
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
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                                    "url": image_url
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
                content = content.replace('```json', '').replace('```', '').strip()

            result = json.loads(content)

            # Validate output
            if result and 'product_name' in result and 'keywords' in result and 'description' in result:
                return result
            else:
                print(f"⚠️ LLM output thiếu fields cho {image_url}")
                return None

        except json.JSONDecodeError as e:
            print(f"⚠️ Lỗi parse JSON từ LLM cho {image_url}: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Lỗi khi phân tích ảnh {image_url}: {e}")
            return None

    def _create_description_text(self, llm_output: Dict) -> str:
        """
        Tạo text mô tả từ LLM output.
        Sử dụng CÙNG logic như ProductVisualSearch.create_search_query().

        Args:
            llm_output: Dictionary từ _analyze_image_from_url()

        Returns:
            Text mô tả để embedding
        """
        product_name = llm_output.get('product_name', '')
        description = llm_output.get('description', '')
        keywords = llm_output.get('keywords', [])

        # Kết hợp: product_name + description + top keywords
        parts = [product_name, description]

        # Thêm top 5 keywords
        if keywords:
            parts.append(' '.join(keywords[:5]))

        text = ' '.join(parts)
        return text

    def _create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embeddings cho nhiều texts (batch).

        Args:
            texts: List các văn bản

        Returns:
            numpy array shape (len(texts), dimension)
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def _save_index(self):
        """Lưu FAISS index và metadata ra disk."""
        if self.index is None:
            print("⚠️ Không có index để lưu")
            return

        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Lưu FAISS index
        faiss.write_index(self.index, self.index_path)

        # Lưu metadata
        metadata = {
            'product_ids': self.product_ids,
            'titles': self.titles
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"💾 Saved index to {self.index_path}")
        print(f"💾 Saved metadata to {self.metadata_path}")

    def _load_index(self):
        """Load FAISS index và metadata từ disk."""
        print("📂 Loading existing product index...")

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.product_ids = metadata['product_ids']
            self.titles = metadata['titles']

        print(f"✅ Loaded product index with {self.index.ntotal} vectors")

    def _load_index_if_exists(self):
        """Load index nếu file tồn tại."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self._load_index()
            except Exception as e:
                print(f"⚠️ Error loading product index: {e}")
                print("💡 Sẽ build index khi gọi update_index()")

    def vectorize_products(self, force_rebuild: bool = False):
        """
        Vector hóa products và lưu vào bảng product_vectors.

        Flow mới:
        1. Lấy ảnh nền (ANH_NEN) của products từ bảng images
        2. Phân tích ảnh bằng LLM Vision → text mô tả
        3. Embedding text mô tả
        4. Lưu vào product_vectors

        Args:
            force_rebuild: Nếu True, vector hóa lại tất cả products.
                          Nếu False, chỉ vector hóa products mới (chưa có trong product_vectors).
        """
        connection = None
        try:
            connection = self._get_db_connection()
            cursor = connection.cursor(dictionary=True)

            if force_rebuild:
                # Vector hóa tất cả products (rebuild toàn bộ)
                print("🔄 Force rebuild - Vector hóa tất cả products...")

                # Query để lấy products và ảnh nền
                query = """
                    SELECT p.id, p.title, i.src as image_url
                    FROM products p
                    LEFT JOIN images i ON p.id = i.product_id AND i.name = 'ANH_NEN'
                """
                cursor.execute(query)
                products = cursor.fetchall()

                if not products:
                    print("⚠️ Không có products nào trong database!")
                    return

                print(f"📚 Tìm thấy {len(products)} products")

                # Phân tích ảnh và tạo embeddings
                product_ids = []
                embeddings_list = []
                success_count = 0
                fallback_count = 0

                for idx, product in enumerate(products, 1):
                    product_id = product['id']
                    title = product['title']
                    image_url = product.get('image_url')

                    print(f"🔄 [{idx}/{len(products)}] Processing product {product_id}...", end=' ')

                    # Nếu có ảnh nền, phân tích bằng LLM
                    text_to_embed = None
                    if image_url:
                        llm_output = self._analyze_image_from_url(image_url)
                        if llm_output:
                            text_to_embed = self._create_description_text(llm_output)
                            print(f"✅ LLM: {llm_output['product_name'][:50]}")
                            success_count += 1
                        else:
                            print(f"⚠️ LLM failed, fallback to title")
                            text_to_embed = title
                            fallback_count += 1
                    else:
                        print(f"⚠️ No image, fallback to title")
                        text_to_embed = title
                        fallback_count += 1

                    # Tạo embedding
                    embedding = self._create_embeddings_batch([text_to_embed])[0]

                    product_ids.append(product_id)
                    embeddings_list.append(embedding)

                # Xóa tất cả vectors cũ
                cursor.execute("DELETE FROM product_vectors")

                # Insert vectors mới
                insert_query = """
                    INSERT INTO product_vectors (product_id, vector)
                    VALUES (%s, %s)
                """
                for product_id, embedding in zip(product_ids, embeddings_list):
                    vector_json = json.dumps(embedding.tolist())
                    cursor.execute(insert_query, (product_id, vector_json))

                connection.commit()
                print(f"\n✅ Đã vector hóa và lưu {len(products)} products")
                print(f"   - LLM Vision: {success_count}")
                print(f"   - Fallback (title): {fallback_count}")

            else:
                # Chỉ vector hóa products mới
                print("🔄 Incremental update - Chỉ vector hóa products mới...")

                query = """
                    SELECT p.id, p.title, i.src as image_url
                    FROM products p
                    LEFT JOIN product_vectors pv ON p.id = pv.product_id
                    LEFT JOIN images i ON p.id = i.product_id AND i.name = 'ANH_NEN'
                    WHERE pv.id IS NULL
                """
                cursor.execute(query)
                new_products = cursor.fetchall()

                if new_products:
                    print(f"📚 Tìm thấy {len(new_products)} products mới")

                    # Phân tích ảnh và tạo embeddings
                    product_ids = []
                    embeddings_list = []
                    success_count = 0
                    fallback_count = 0

                    for idx, product in enumerate(new_products, 1):
                        product_id = product['id']
                        title = product['title']
                        image_url = product.get('image_url')

                        print(f"🔄 [{idx}/{len(new_products)}] Processing product {product_id}...", end=' ')

                        # Nếu có ảnh nền, phân tích bằng LLM
                        text_to_embed = None
                        if image_url:
                            llm_output = self._analyze_image_from_url(image_url)
                            if llm_output:
                                text_to_embed = self._create_description_text(llm_output)
                                print(f"✅ LLM: {llm_output['product_name'][:50]}")
                                success_count += 1
                            else:
                                print(f"⚠️ LLM failed, fallback to title")
                                text_to_embed = title
                                fallback_count += 1
                        else:
                            print(f"⚠️ No image, fallback to title")
                            text_to_embed = title
                            fallback_count += 1

                        # Tạo embedding
                        embedding = self._create_embeddings_batch([text_to_embed])[0]

                        product_ids.append(product_id)
                        embeddings_list.append(embedding)

                    # Insert vectors
                    insert_query = """
                        INSERT INTO product_vectors (product_id, vector)
                        VALUES (%s, %s)
                    """
                    for product_id, embedding in zip(product_ids, embeddings_list):
                        vector_json = json.dumps(embedding.tolist())
                        cursor.execute(insert_query, (product_id, vector_json))

                    connection.commit()
                    print(f"\n✅ Đã vector hóa và lưu {len(new_products)} products mới")
                    print(f"   - LLM Vision: {success_count}")
                    print(f"   - Fallback (title): {fallback_count}")
                else:
                    print("✅ Không có products mới nào cần vector hóa")

        except Error as e:
            print(f"❌ Lỗi khi vector hóa products: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    def build_index(self):
        """
        Build FAISS index từ bảng product_vectors.
        """
        connection = None
        try:
            print("🔨 Building FAISS index từ product_vectors...")

            connection = self._get_db_connection()
            cursor = connection.cursor(dictionary=True)

            # Lấy tất cả vectors từ database
            query = """
                SELECT pv.product_id, p.title, pv.vector
                FROM product_vectors pv
                JOIN products p ON pv.product_id = p.id
                ORDER BY pv.product_id
            """
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                print("⚠️ Không có vectors nào trong database!")
                print("💡 Hãy chạy vectorize_products() trước")
                return

            print(f"📚 Loaded {len(results)} vectors từ database")

            # Parse vectors từ JSON
            vectors = []
            self.product_ids = []
            self.titles = []

            for row in results:
                vector_list = json.loads(row['vector'])
                vectors.append(vector_list)
                self.product_ids.append(row['product_id'])
                self.titles.append(row['title'])

            # Convert sang numpy array
            vectors_array = np.array(vectors, dtype=np.float32)

            # Tạo FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors_array)

            print(f"✅ Index built với {self.index.ntotal} vectors")

            # Lưu index và metadata ra file
            self._save_index()

        except Error as e:
            print(f"❌ Lỗi khi build index: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    def update_index(self, force_rebuild: bool = False):
        """
        Update FAISS index (vector hóa products + rebuild index).

        Args:
            force_rebuild: Nếu True, vector hóa lại tất cả products (dùng cho /build-product-index)
                          Nếu False, chỉ vector hóa products mới (dùng cho /update-product-index)
        """
        print("=" * 60)
        if force_rebuild:
            print("🔨 BUILDING PRODUCT INDEX (Force Rebuild)")
        else:
            print("🔄 UPDATING PRODUCT INDEX (Incremental)")
        print("=" * 60)

        # Bước 1: Vector hóa products
        self.vectorize_products(force_rebuild=force_rebuild)

        # Bước 2: Rebuild index
        self.build_index()

        print("=" * 60)
        print("✅ Index updated successfully!")
        print("=" * 60)

    def get_stats(self) -> Dict:
        """Lấy thống kê về index."""
        connection = None
        try:
            connection = self._get_db_connection()
            cursor = connection.cursor()

            # Đếm số products
            cursor.execute("SELECT COUNT(*) FROM products")
            total_products = cursor.fetchone()[0]

            # Đếm số vectors
            cursor.execute("SELECT COUNT(*) FROM product_vectors")
            total_vectors = cursor.fetchone()[0]

            stats = {
                'total_products': total_products,
                'total_vectors': total_vectors,
                'vectorized_percentage': round((total_vectors / total_products * 100), 2) if total_products > 0 else 0,
                'dimension': self.dimension,
                'model': self.embedding_model
            }

            if self.index:
                stats['index_size'] = self.index.ntotal

            return stats

        except Error as e:
            print(f"❌ Lỗi khi lấy stats: {e}")
            return {}
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()
