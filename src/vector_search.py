"""
Vector Search module sử dụng FAISS và OpenAI Embeddings.
Implement RAG (Retrieval-Augmented Generation) cho chatbot FAQ.
"""

import json
import pickle
import os
from typing import List, Dict, Tuple
import numpy as np
import faiss
from openai import OpenAI


class VectorSearch:
    """
    Quản lý vector search với FAISS index và OpenAI Embeddings.

    Flow:
    1. Build index: Encode tất cả FAQ → FAISS index (1 lần)
    2. Search: Encode query → FAISS search → Top-k similar FAQs
    """

    def __init__(
        self,
        kb_path: str,
        index_path: str = "data/faiss_index.bin",
        metadata_path: str = "data/metadata.pkl",
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        """
        Khởi tạo VectorSearch.

        Args:
            kb_path: Đường dẫn tới knowledge_base.json
            index_path: Đường dẫn lưu FAISS index
            metadata_path: Đường dẫn lưu metadata (questions, answers)
            embedding_model: Model OpenAI Embeddings
            dimension: Số chiều của vector (1536 cho text-embedding-3-small)
        """
        self.kb_path = kb_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = embedding_model
        self.dimension = dimension

        # Khởi tạo OpenAI client
        self.client = OpenAI()

        # Load hoặc build index
        self.index, self.metadata = self._load_or_build_index()

    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Tạo embedding vector từ text sử dụng OpenAI API.

        Args:
            text: Văn bản cần encode

        Returns:
            numpy array shape (dimension,)
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

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

    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base từ JSON."""
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_index(self) -> Tuple[faiss.Index, Dict]:
        """
        Build FAISS index từ knowledge base.

        Returns:
            (FAISS index, metadata dict)
        """
        print("🔨 Building FAISS index...")

        # Load KB
        kb = self._load_knowledge_base()

        # Chuẩn bị data
        questions = []
        answers = []
        categories = []

        for category in kb['categories']:
            category_name = category['name']
            for qa in category['questions']:
                questions.append(qa['question'])
                answers.append(qa['answer'])
                categories.append(category_name)

        print(f"📚 Loaded {len(questions)} FAQs from {len(kb['categories'])} categories")

        # Tạo embeddings cho tất cả câu hỏi (batch để nhanh hơn)
        print("🔄 Creating embeddings...")
        embeddings = self._create_embeddings_batch(questions)

        # Tạo FAISS index
        # IndexFlatL2: Simple, exact search với L2 distance
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)

        print(f"✅ Index built with {index.ntotal} vectors")

        # Metadata
        metadata = {
            'questions': questions,
            'answers': answers,
            'categories': categories,
            'total': len(questions)
        }

        # Lưu index và metadata
        self._save_index(index, metadata)

        return index, metadata

    def _save_index(self, index: faiss.Index, metadata: Dict):
        """Lưu FAISS index và metadata ra disk."""
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Lưu FAISS index
        faiss.write_index(index, self.index_path)

        # Lưu metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"💾 Saved index to {self.index_path}")
        print(f"💾 Saved metadata to {self.metadata_path}")

    def _load_index(self) -> Tuple[faiss.Index, Dict]:
        """Load FAISS index và metadata từ disk."""
        print("📂 Loading existing index...")

        # Load FAISS index
        index = faiss.read_index(self.index_path)

        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        print(f"✅ Loaded index with {index.ntotal} vectors")

        return index, metadata

    def _load_or_build_index(self) -> Tuple[faiss.Index, Dict]:
        """Load index nếu có, không thì build mới."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                return self._load_index()
            except Exception as e:
                print(f"⚠️ Error loading index: {e}")
                print("🔨 Rebuilding index...")
                return self._build_index()
        else:
            return self._build_index()

    def rebuild_index(self):
        """Force rebuild index (khi update knowledge base)."""
        self.index, self.metadata = self._build_index()

    def search(
        self,
        query: str,
        top_k: int = 3,
        distance_threshold: float = None
    ) -> List[Dict]:
        """
        Tìm kiếm FAQs tương tự với query.

        Args:
            query: Câu hỏi từ user
            top_k: Số lượng kết quả trả về
            distance_threshold: Ngưỡng khoảng cách tối đa (None = không filter)

        Returns:
            List các FAQ với format:
            [
                {
                    'question': str,
                    'answer': str,
                    'category': str,
                    'distance': float,  # L2 distance (càng nhỏ càng giống)
                    'similarity': float  # 0-1 (càng cao càng giống)
                }
            ]
        """
        # Encode query
        query_vector = self._create_embedding(query)
        query_vector = query_vector.reshape(1, -1)  # Shape: (1, dimension)

        # Search trong FAISS
        distances, indices = self.index.search(query_vector, top_k)

        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # Filter theo threshold nếu có
            if distance_threshold is not None and dist > distance_threshold:
                continue

            # Convert L2 distance sang similarity score (0-1)
            # Distance càng nhỏ → similarity càng cao
            # Dùng công thức: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + dist)

            results.append({
                'question': self.metadata['questions'][idx],
                'answer': self.metadata['answers'][idx],
                'category': self.metadata['categories'][idx],
                'distance': float(dist),
                'similarity': float(similarity),
                'rank': i + 1
            })

        return results

    def get_stats(self) -> Dict:
        """Lấy thống kê về index."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'model': self.embedding_model,
            'total_questions': self.metadata['total'],
            'categories': list(set(self.metadata['categories']))
        }


if __name__ == "__main__":
    # Test code
    vs = VectorSearch(kb_path="data/knowledge_base.json")

    # Test search
    test_queries = [
        "Giao hàng mất bao lâu?",
        "Bao giờ tôi nhận được đơn?",
        "Ship có mất phí không?",
        "Làm sao để theo dõi đơn hàng?"
    ]

    print("\n" + "="*60)
    print("TEST VECTOR SEARCH")
    print("="*60)

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vs.search(query, top_k=2)

        for r in results:
            print(f"  [{r['rank']}] Similarity: {r['similarity']:.3f}")
            print(f"      Q: {r['question']}")
            print(f"      Category: {r['category']}")
