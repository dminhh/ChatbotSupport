"""
Product Vector Indexer - Vector hóa products và build FAISS index.
Lưu vectors vào bảng product_vectors trong MySQL.
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

                cursor.execute("SELECT id, title FROM products")
                products = cursor.fetchall()

                if not products:
                    print("⚠️ Không có products nào trong database!")
                    return

                print(f"📚 Tìm thấy {len(products)} products")

                # Vector hóa batch
                product_ids = [p['id'] for p in products]
                titles = [p['title'] for p in products]

                print("🔄 Đang tạo embeddings...")
                embeddings = self._create_embeddings_batch(titles)

                # Xóa tất cả vectors cũ
                cursor.execute("DELETE FROM product_vectors")

                # Insert vectors mới
                insert_query = """
                    INSERT INTO product_vectors (product_id, vector)
                    VALUES (%s, %s)
                """
                for product_id, embedding in zip(product_ids, embeddings):
                    vector_json = json.dumps(embedding.tolist())
                    cursor.execute(insert_query, (product_id, vector_json))

                connection.commit()
                print(f"✅ Đã vector hóa và lưu {len(products)} products")

            else:
                # Chỉ vector hóa products mới
                print("🔄 Incremental update - Chỉ vector hóa products mới...")

                query = """
                    SELECT p.id, p.title
                    FROM products p
                    LEFT JOIN product_vectors pv ON p.id = pv.product_id
                    WHERE pv.id IS NULL
                """
                cursor.execute(query)
                new_products = cursor.fetchall()

                if new_products:
                    print(f"📚 Tìm thấy {len(new_products)} products mới")

                    product_ids = [p['id'] for p in new_products]
                    titles = [p['title'] for p in new_products]

                    print("🔄 Đang tạo embeddings...")
                    embeddings = self._create_embeddings_batch(titles)

                    # Insert vectors
                    insert_query = """
                        INSERT INTO product_vectors (product_id, vector)
                        VALUES (%s, %s)
                    """
                    for product_id, embedding in zip(product_ids, embeddings):
                        vector_json = json.dumps(embedding.tolist())
                        cursor.execute(insert_query, (product_id, vector_json))

                    connection.commit()
                    print(f"✅ Đã vector hóa và lưu {len(new_products)} products mới")
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
