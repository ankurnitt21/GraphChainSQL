-- Warehouse Management Schema + Seed Data

-- === WAREHOUSE & STORAGE ===
CREATE TABLE warehouse (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    city VARCHAR(50),
    state VARCHAR(50),
    capacity_sqft NUMERIC(12,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE zone (
    id BIGSERIAL PRIMARY KEY,
    warehouse_id BIGINT NOT NULL REFERENCES warehouse(id),
    code VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    zone_type VARCHAR(30) NOT NULL,
    temperature_controlled BOOLEAN DEFAULT FALSE,
    max_capacity_units INTEGER,
    utilization_pct NUMERIC(5,2) DEFAULT 0,
    UNIQUE(warehouse_id, code)
);

CREATE TABLE location (
    id BIGSERIAL PRIMARY KEY,
    zone_id BIGINT NOT NULL REFERENCES zone(id),
    aisle VARCHAR(10),
    rack VARCHAR(10),
    shelf VARCHAR(10),
    bin VARCHAR(10),
    barcode VARCHAR(50) UNIQUE,
    location_type VARCHAR(30) NOT NULL,
    max_weight_kg NUMERIC(10,2),
    is_occupied BOOLEAN DEFAULT FALSE
);

-- === PRODUCT & CATALOG ===
CREATE TABLE category (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_category_id BIGINT REFERENCES category(id),
    description TEXT
);

CREATE TABLE supplier (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(150) NOT NULL,
    contact_name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(30),
    city VARCHAR(50),
    country VARCHAR(50),
    lead_time_days INTEGER DEFAULT 7,
    rating NUMERIC(3,2)
);

CREATE TABLE product (
    id BIGSERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category_id BIGINT REFERENCES category(id),
    supplier_id BIGINT REFERENCES supplier(id),
    unit_price NUMERIC(12,2),
    cost_price NUMERIC(12,2),
    weight_kg NUMERIC(10,3),
    uom VARCHAR(20) DEFAULT 'EACH',
    is_perishable BOOLEAN DEFAULT FALSE,
    min_stock_level INTEGER DEFAULT 0,
    reorder_point INTEGER DEFAULT 0,
    reorder_qty INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- === INVENTORY ===
CREATE TABLE inventory (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES product(id),
    location_id BIGINT NOT NULL REFERENCES location(id),
    quantity_on_hand INTEGER NOT NULL DEFAULT 0,
    quantity_reserved INTEGER NOT NULL DEFAULT 0,
    quantity_available INTEGER GENERATED ALWAYS AS (quantity_on_hand - quantity_reserved) STORED,
    lot_number VARCHAR(50),
    expiry_date DATE,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(product_id, location_id, lot_number)
);

CREATE TABLE inventory_transaction (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES product(id),
    from_location_id BIGINT REFERENCES location(id),
    to_location_id BIGINT REFERENCES location(id),
    transaction_type VARCHAR(30) NOT NULL,
    quantity INTEGER NOT NULL,
    reference_type VARCHAR(30),
    reference_id BIGINT,
    performed_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- === PROCUREMENT ===
CREATE TABLE purchase_order (
    id BIGSERIAL PRIMARY KEY,
    po_number VARCHAR(30) UNIQUE NOT NULL,
    supplier_id BIGINT NOT NULL REFERENCES supplier(id),
    warehouse_id BIGINT NOT NULL REFERENCES warehouse(id),
    status VARCHAR(20) DEFAULT 'DRAFT',
    order_date DATE,
    expected_delivery DATE,
    total_amount NUMERIC(14,2),
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE purchase_order_line (
    id BIGSERIAL PRIMARY KEY,
    purchase_order_id BIGINT NOT NULL REFERENCES purchase_order(id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL REFERENCES product(id),
    quantity_ordered INTEGER NOT NULL,
    quantity_received INTEGER DEFAULT 0,
    unit_price NUMERIC(12,2)
);

-- === SALES & FULFILLMENT ===
CREATE TABLE customer (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(30),
    city VARCHAR(50),
    state VARCHAR(50),
    customer_type VARCHAR(30),
    credit_limit NUMERIC(14,2),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE sales_order (
    id BIGSERIAL PRIMARY KEY,
    order_number VARCHAR(30) UNIQUE NOT NULL,
    customer_id BIGINT NOT NULL REFERENCES customer(id),
    warehouse_id BIGINT NOT NULL REFERENCES warehouse(id),
    status VARCHAR(20) DEFAULT 'PENDING',
    priority VARCHAR(10) DEFAULT 'NORMAL',
    order_date DATE,
    required_date DATE,
    total_amount NUMERIC(14,2),
    shipping_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE sales_order_line (
    id BIGSERIAL PRIMARY KEY,
    sales_order_id BIGINT NOT NULL REFERENCES sales_order(id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL REFERENCES product(id),
    quantity_ordered INTEGER NOT NULL,
    quantity_shipped INTEGER DEFAULT 0,
    unit_price NUMERIC(12,2)
);

CREATE TABLE shipment (
    id BIGSERIAL PRIMARY KEY,
    shipment_number VARCHAR(30) UNIQUE NOT NULL,
    sales_order_id BIGINT NOT NULL REFERENCES sales_order(id),
    carrier VARCHAR(100),
    tracking_number VARCHAR(100),
    status VARCHAR(20) DEFAULT 'PENDING',
    shipped_date TIMESTAMP,
    delivered_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- === APP TABLES ===
CREATE TABLE schema_description (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100),
    domain VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    data_type VARCHAR(50),
    embedding_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE conversation (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sql_query TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE conversation_summary (
    session_id VARCHAR(100) PRIMARY KEY,
    summary TEXT NOT NULL,
    approximate_tokens INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE query_feedback (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    run_id VARCHAR(100),
    query TEXT NOT NULL,
    generated_sql TEXT,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    correction TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_conv_session ON conversation(session_id);
CREATE INDEX idx_schema_domain ON schema_description(domain);
CREATE INDEX idx_feedback_session ON query_feedback(session_id);
CREATE INDEX idx_feedback_rating ON query_feedback(rating);

-- Full-text search index for schema retrieval
CREATE INDEX idx_schema_fts ON schema_description
    USING gin(to_tsvector('english', description || ' ' || table_name || ' ' || COALESCE(column_name, '')));

-- === SCHEMA DESCRIPTIONS (for hybrid retrieval) ===
INSERT INTO schema_description (table_name, column_name, domain, description, data_type) VALUES
('warehouse', NULL, 'warehouse', 'Physical warehouse locations with capacity and active status', NULL),
('warehouse', 'code', 'warehouse', 'Unique warehouse identifier code like WH-EAST-01', 'VARCHAR(20)'),
('warehouse', 'capacity_sqft', 'warehouse', 'Total warehouse capacity in square feet', 'NUMERIC(12,2)'),
('zone', NULL, 'warehouse', 'Storage zones within warehouses: RECEIVING, STORAGE, PICKING, SHIPPING, COLD_STORAGE, HAZMAT', NULL),
('zone', 'utilization_pct', 'warehouse', 'Current utilization percentage of the zone', 'NUMERIC(5,2)'),
('location', NULL, 'warehouse', 'Physical storage locations (bins, racks, shelves) within zones', NULL),
('product', NULL, 'product', 'Product catalog with SKU, pricing, weight, and reorder settings', NULL),
('product', 'unit_price', 'product', 'Selling price per unit of the product', 'NUMERIC(12,2)'),
('product', 'cost_price', 'product', 'Purchase/cost price from supplier', 'NUMERIC(12,2)'),
('product', 'reorder_point', 'product', 'Inventory level that triggers reorder', 'INTEGER'),
('category', NULL, 'product', 'Product categories with hierarchical parent-child relationships', NULL),
('supplier', NULL, 'procurement', 'Suppliers with contact info, lead time, and performance rating', NULL),
('supplier', 'lead_time_days', 'procurement', 'Average days for supplier to deliver orders', 'INTEGER'),
('supplier', 'rating', 'procurement', 'Supplier performance rating from 0.00 to 5.00', 'NUMERIC(3,2)'),
('inventory', NULL, 'inventory', 'Current stock levels per product per location with lot tracking', NULL),
('inventory', 'quantity_on_hand', 'inventory', 'Total physical quantity in stock at location', 'INTEGER'),
('inventory', 'quantity_reserved', 'inventory', 'Quantity reserved for pending orders', 'INTEGER'),
('inventory', 'quantity_available', 'inventory', 'Available quantity (on_hand - reserved), computed column', 'INTEGER'),
('inventory_transaction', NULL, 'inventory', 'Audit trail of all inventory movements: RECEIPT, PICK, TRANSFER, ADJUSTMENT, RETURN', NULL),
('purchase_order', NULL, 'procurement', 'Purchase orders to suppliers with status: DRAFT, SUBMITTED, CONFIRMED, RECEIVED, CANCELLED', NULL),
('purchase_order', 'total_amount', 'procurement', 'Total monetary value of the purchase order', 'NUMERIC(14,2)'),
('purchase_order_line', NULL, 'procurement', 'Line items in purchase orders with ordered vs received quantities', NULL),
('customer', NULL, 'sales', 'Customer accounts: RETAIL, WHOLESALE, DISTRIBUTOR with credit limits', NULL),
('sales_order', NULL, 'sales', 'Customer sales orders with status: PENDING, CONFIRMED, PICKING, SHIPPED, DELIVERED, CANCELLED', NULL),
('sales_order', 'priority', 'sales', 'Order priority: LOW, NORMAL, HIGH, URGENT', 'VARCHAR(10)'),
('sales_order', 'total_amount', 'sales', 'Total monetary value of the sales order', 'NUMERIC(14,2)'),
('sales_order_line', NULL, 'sales', 'Line items in sales orders with ordered vs shipped quantities', NULL),
('shipment', NULL, 'sales', 'Shipment tracking for sales orders with carrier and delivery info', NULL),
('shipment', 'status', 'sales', 'Shipment status: PENDING, IN_TRANSIT, DELIVERED', 'VARCHAR(20)');

-- === SEED DATA ===
INSERT INTO warehouse (code, name, city, state, capacity_sqft) VALUES
('WH-EAST-01', 'East Coast Distribution Center', 'Newark', 'NJ', 250000),
('WH-WEST-01', 'West Coast Fulfillment Hub', 'Ontario', 'CA', 320000),
('WH-CENT-01', 'Central Regional Warehouse', 'Dallas', 'TX', 180000);

INSERT INTO zone (warehouse_id, code, name, zone_type, max_capacity_units, utilization_pct) VALUES
(1, 'RCV-01', 'Receiving Dock', 'RECEIVING', 5000, 45),
(1, 'STR-01', 'General Storage A', 'STORAGE', 50000, 72),
(1, 'STR-02', 'Cold Storage', 'COLD_STORAGE', 10000, 55),
(1, 'PCK-01', 'Picking Zone', 'PICKING', 8000, 60),
(1, 'SHP-01', 'Shipping Dock', 'SHIPPING', 3000, 35),
(2, 'STR-01', 'Main Storage', 'STORAGE', 80000, 78),
(2, 'PCK-01', 'Pick & Pack', 'PICKING', 12000, 65),
(3, 'STR-01', 'Primary Storage', 'STORAGE', 40000, 82);

INSERT INTO location (zone_id, aisle, rack, shelf, bin, barcode, location_type, max_weight_kg, is_occupied) VALUES
(2, 'A', '01', '01', '01', 'LOC-A010101', 'RACK', 500, TRUE),
(2, 'A', '01', '02', '01', 'LOC-A010201', 'RACK', 500, TRUE),
(2, 'A', '02', '01', '01', 'LOC-A020101', 'RACK', 500, FALSE),
(2, 'B', '01', '01', '01', 'LOC-B010101', 'BULK', 2000, TRUE),
(3, 'F', '01', '01', '01', 'LOC-F010101', 'RACK', 400, TRUE),
(4, 'P', '01', '01', '01', 'LOC-P010101', 'PICK_FACE', 200, TRUE),
(6, 'A', '01', '01', '01', 'LOC-W-A0101', 'RACK', 600, TRUE),
(6, 'B', '01', '01', '01', 'LOC-W-B0101', 'PALLET', 1200, TRUE),
(8, 'A', '01', '01', '01', 'LOC-C-A0101', 'RACK', 500, TRUE);

INSERT INTO category (name, description) VALUES
('Electronics', 'Electronic devices and accessories'),
('Furniture', 'Office and warehouse furniture'),
('Safety Equipment', 'PPE and safety gear'),
('Packaging', 'Boxes, wrap, and shipping supplies'),
('Food & Beverage', 'Perishable and non-perishable items');

INSERT INTO supplier (code, name, contact_name, email, city, country, lead_time_days, rating) VALUES
('SUP-TECH', 'TechWave Electronics', 'David Park', 'david@techwave.com', 'San Jose', 'USA', 5, 4.50),
('SUP-FURN', 'OfficeMax Furniture', 'Rachel Green', 'rachel@officemax.com', 'Grand Rapids', 'USA', 14, 4.20),
('SUP-SAFE', 'SafeGuard Industries', 'Tom Bradley', 'tom@safeguard.com', 'Cincinnati', 'USA', 7, 4.80),
('SUP-PACK', 'PackRight Solutions', 'Nina Patel', 'nina@packright.com', 'Memphis', 'USA', 3, 4.60),
('SUP-FOOD', 'FreshChain Foods', 'Amy Wu', 'amy@freshchain.com', 'Portland', 'USA', 2, 4.70);

INSERT INTO product (sku, name, description, category_id, supplier_id, unit_price, cost_price, weight_kg, uom, min_stock_level, reorder_point, reorder_qty) VALUES
('SKU-LAPTOP-001', 'ProBook Laptop 15"', 'Business laptop 16GB RAM', 1, 1, 899.99, 650, 2.1, 'EACH', 50, 100, 200),
('SKU-TABLET-001', 'SmartTab Pro 10"', '10-inch tablet with stylus', 1, 1, 499.99, 320, 0.55, 'EACH', 30, 60, 100),
('SKU-CABLE-001', 'Cat6 Ethernet Cable 10ft', 'Network cable', 1, 1, 12.99, 4.5, 0.15, 'EACH', 200, 500, 1000),
('SKU-DESK-001', 'Ergonomic Standing Desk', 'Electric height-adjustable', 2, 2, 549.99, 320, 35, 'EACH', 10, 20, 50),
('SKU-CHAIR-001', 'Executive Mesh Chair', 'Lumbar support chair', 2, 2, 349.99, 180, 18, 'EACH', 15, 30, 60),
('SKU-HHAT-001', 'Hard Hat Type II White', 'ANSI certified', 3, 3, 24.99, 8.5, 0.4, 'EACH', 100, 200, 500),
('SKU-VEST-001', 'Hi-Vis Safety Vest', 'Class 2 reflective', 3, 3, 14.99, 5, 0.2, 'EACH', 150, 300, 600),
('SKU-BOX-SM', 'Shipping Box Small', '12x10x8 corrugated', 4, 4, 1.99, 0.45, 0.3, 'EACH', 500, 1000, 3000),
('SKU-BOX-LG', 'Shipping Box Large', '24x18x18 heavy-duty', 4, 4, 5.99, 1.5, 0.8, 'EACH', 200, 500, 1500),
('SKU-CHICKEN', 'Frozen Chicken Breast 5lb', 'Boneless skinless', 5, 5, 12.99, 7.5, 2.27, 'EACH', 100, 200, 500);

INSERT INTO inventory (product_id, location_id, quantity_on_hand, quantity_reserved, lot_number) VALUES
(1, 1, 120, 15, 'LOT-2024-001'),
(1, 2, 80, 10, 'LOT-2024-002'),
(2, 2, 200, 25, 'LOT-2024-003'),
(3, 4, 1500, 200, 'LOT-2024-004'),
(4, 4, 25, 3, 'LOT-2024-005'),
(5, 4, 40, 5, 'LOT-2024-006'),
(6, 6, 450, 50, 'LOT-2024-007'),
(7, 6, 600, 75, 'LOT-2024-008'),
(8, 6, 3000, 500, 'LOT-2024-009'),
(10, 5, 300, 40, 'LOT-FRZ-001'),
(1, 7, 90, 8, 'LOT-2024-010'),
(3, 8, 2500, 300, 'LOT-2024-011'),
(9, 9, 180, 25, 'LOT-2024-012');

INSERT INTO customer (code, name, email, city, state, customer_type, credit_limit) VALUES
('CUST-001', 'Metro Office Supplies', 'john@metrooffice.com', 'New York', 'NY', 'RETAIL', 50000),
('CUST-002', 'BulkBuy Distributors', 'maria@bulkbuy.com', 'Chicago', 'IL', 'WHOLESALE', 250000),
('CUST-003', 'Pacific Coast Trading', 'kevin@pctrade.com', 'Long Beach', 'CA', 'DISTRIBUTOR', 500000),
('CUST-004', 'GreenTech Solutions', 'priya@greentech.com', 'Austin', 'TX', 'RETAIL', 75000);

INSERT INTO purchase_order (po_number, supplier_id, warehouse_id, status, order_date, expected_delivery, total_amount, created_by) VALUES
('PO-2024-001', 1, 1, 'RECEIVED', '2024-08-01', '2024-08-10', 175000, 'admin'),
('PO-2024-002', 4, 1, 'RECEIVED', '2024-09-15', '2024-09-20', 12500, 'admin'),
('PO-2024-003', 5, 1, 'CONFIRMED', '2024-10-15', '2024-10-20', 8500, 'admin'),
('PO-2024-004', 2, 2, 'SUBMITTED', '2024-11-05', '2024-11-25', 45000, 'admin');

INSERT INTO purchase_order_line (purchase_order_id, product_id, quantity_ordered, quantity_received, unit_price) VALUES
(1, 1, 200, 200, 650), (1, 2, 100, 100, 320),
(2, 8, 3000, 3000, 0.45), (2, 9, 1000, 1000, 1.5),
(3, 10, 500, 300, 7.5), (4, 4, 50, 0, 320);

INSERT INTO sales_order (order_number, customer_id, warehouse_id, status, priority, order_date, required_date, total_amount, shipping_method) VALUES
('SO-2024-001', 1, 1, 'DELIVERED', 'NORMAL', '2024-09-20', '2024-09-28', 4500, 'UPS Ground'),
('SO-2024-002', 2, 1, 'SHIPPED', 'HIGH', '2024-10-01', '2024-10-05', 32500, 'FedEx Freight'),
('SO-2024-003', 3, 2, 'PICKING', 'URGENT', '2024-11-01', '2024-11-04', 55000, 'FedEx Express'),
('SO-2024-004', 1, 1, 'PENDING', 'NORMAL', '2024-11-08', '2024-11-15', 2100, 'USPS Priority');

INSERT INTO sales_order_line (sales_order_id, product_id, quantity_ordered, quantity_shipped, unit_price) VALUES
(1, 1, 5, 5, 899.99), (2, 1, 25, 25, 899.99), (2, 3, 500, 500, 12.99),
(3, 1, 50, 0, 899.99), (3, 2, 30, 0, 499.99), (4, 6, 50, 0, 24.99);

INSERT INTO shipment (shipment_number, sales_order_id, carrier, tracking_number, status, shipped_date, delivered_date) VALUES
('SHP-001', 1, 'UPS', '1Z999AA10123456784', 'DELIVERED', '2024-09-25', '2024-09-27'),
('SHP-002', 2, 'FedEx', 'FX7890123456', 'IN_TRANSIT', '2024-10-03', NULL);

INSERT INTO inventory_transaction (product_id, from_location_id, to_location_id, transaction_type, quantity, reference_type, reference_id, performed_by, created_at) VALUES
(1, NULL, 1, 'RECEIPT', 120, 'PURCHASE_ORDER', 1, 'warehouse_ops', '2024-08-09 08:00:00'),
(1, 1, NULL, 'PICK', 5, 'SALES_ORDER', 1, 'picker_01', '2024-09-24 10:00:00'),
(1, 1, NULL, 'PICK', 25, 'SALES_ORDER', 2, 'picker_01', '2024-10-02 08:00:00'),
(3, NULL, 4, 'RECEIPT', 1500, 'PURCHASE_ORDER', 1, 'warehouse_ops', '2024-08-09 09:00:00'),
(8, NULL, 6, 'ADJUSTMENT', -50, 'ADJUSTMENT', NULL, 'supervisor', '2024-10-20 14:00:00');
