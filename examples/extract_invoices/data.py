"""Synthetic vendor invoices -- a classic 'pile' shape (docs/design.html §7):
large, independent per-invoice, and checkable against the source text."""

INVOICES: dict[str, str] = {
    "inv-001": """\
Acme Robotics Supply
Invoice #A-1042
Bill to: Example Corp

Items:
1. Servo motor bracket kit — $420.00
2. Cable harness, 3m — $85.50
3. Expedited shipping — $40.00

Total due: $545.50 USD
""",
    "inv-002": """\
Blue Ridge Office Partners
Invoice No: BR-88213
Client: Example Corp, Procurement

Line items:
 - Standing desk (x2) ................ $1,240.00
 - Ergonomic chair (x2) ............... $860.00
 - Delivery & setup ................... $150.00

Amount due: $2,250.00 USD
Payment terms: Net 30
""",
    "inv-003": """\
Northwind Cloud Services
Invoice #: NW-2026-0417
For: Example Corp — March usage

Charges:
1) Compute hours (1,204 hrs) — $602.00
2) Storage (2.1 TB) — $63.00
3) Egress bandwidth — $18.75
4) Support plan (Business tier) — $99.00

TOTAL: $782.75
""",
    "inv-004": """\
Harborview Print & Signage
Invoice #HV-5591

Bill To: Example Corp Marketing Dept

1. Trade show banner, 8ft — $310.00
2. Vinyl decals (qty 50) — $175.00
3. Rush production fee — $60.00
4. Freight — $45.00

Total: $590.00
""",
}
