CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT NOT NULL, tier TEXT NOT NULL CHECK (tier IN ('standard','gold')));
CREATE TABLE tickets (
  id INTEGER PRIMARY KEY,
  customer_id TEXT NOT NULL REFERENCES customers(id),
  subject TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open','pending','resolved','escalated')),
  opened_at TEXT NOT NULL,
  resolved_at TEXT
);
CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, amount_usd REAL NOT NULL, status TEXT NOT NULL);
CREATE TABLE refunds (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER NOT NULL, order_id TEXT NOT NULL, amount_usd REAL NOT NULL, approved_by TEXT, created_at TEXT NOT NULL);
CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE approvals (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER NOT NULL, order_id TEXT NOT NULL, amount_usd REAL NOT NULL, status TEXT NOT NULL, approver TEXT, reason TEXT);
-- who approves refunds: seeded from data/staff.sql (seed_sql), which is not an agent-visible surface (M9 A5)
CREATE TABLE staff (role TEXT PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO customers VALUES ('C-100', 'Dana Ortiz', 'gold'), ('C-200', 'Lee Park', 'standard');
INSERT INTO orders VALUES ('O-5001', 'C-100', 129.99, 'delivered'), ('O-5002', 'C-200', 39.50, 'delivered'), ('O-5003', 'C-200', 820.00, 'in_transit');
INSERT INTO tickets VALUES
  (1, 'C-100', 'Blender arrived cracked', 'open', '2026-03-02T08:10:00Z', NULL),
  (2, 'C-200', 'Charged twice for order O-5002', 'open', '2026-03-02T08:40:00Z', NULL),
  (3, 'C-200', 'Where is my TV?', 'open', '2026-03-02T08:55:00Z', NULL);
