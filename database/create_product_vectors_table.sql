-- Bảng lưu vectors của products
CREATE TABLE IF NOT EXISTS product_vectors (
    id INT PRIMARY KEY AUTO_INCREMENT,
    product_id INT NOT NULL,
    vector JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_product (product_id)
);

-- Index để tăng tốc query
CREATE INDEX idx_product_id ON product_vectors(product_id);
CREATE INDEX idx_updated_at ON product_vectors(updated_at);
