"""
Example table schema JSON format.
Create one JSON file per table in the schemas/ directory.
"""

# This file shows the structure for defining database schemas
# Each schema should be saved as a separate JSON file in schemas/

EXAMPLE_SCHEMA_STRUCTURE = {
    "table_name": "example_table",
    "description": "Brief description of what this table stores",
    "business_context": "Business logic explanation - how this table is used, relationships, important notes",
    "columns": {
        "column_name": {
            "type": "SQL_TYPE",
            "constraints": "PRIMARY KEY, NOT NULL, etc.",
            "description": "What this column represents"
        }
    },
    "relationships": [
        {
            "type": "one_to_many | many_to_one | many_to_many",
            "related_table": "other_table_name",
            "foreign_key": "foreign_key_column",
            "description": "Explanation of the relationship"
        }
    ],
    "indexes": ["column1", "column2"],
    "example_queries": [
        "SELECT * FROM example_table WHERE condition"
    ]
}

# See schemas/ directory for real examples:
# - schemas/customers.json
# - schemas/orders.json
# - schemas/order_items.json
# - schemas/products.json
