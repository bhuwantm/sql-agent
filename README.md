# SQL Agent with RAG

A model-agnostic SQL query generation agent using ChromaDB for retrieval-augmented generation (RAG).

## Features

✅ **Model Agnostic** - Swap between OpenAI, Anthropic, Ollama  
✅ **RAG with ChromaDB** - Only retrieves relevant table schemas  
✅ **Persistent Storage** - ChromaDB data saved to disk  
✅ **Intelligent Change Detection** - Only loads new/modified schemas  
✅ **JSON Schema Format** - One file per table (supports flexible formats)  
✅ **Interactive Mode** - Test queries in real-time  
✅ **Easy to Extend** - Add more providers or features

## Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Configure settings**

Edit `settings.py` to customize paths:

```python
CHROMA_DIRECTORY = "chroma_db"              # Where ChromaDB stores data
CHROMA_COLLECTION_NAME = "database_schemas" # Collection name
DATABASE_SCHEMAS_DIR = "schemas"            # Where schema JSON files are
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. **Create schema files**

   - Add JSON files for your tables in `schemas/` directory (or path from settings)
   - See `examples/example_schemas.py` for format
   - Sample schemas already included: customers, orders, order_items, products

5. **Run the agent**

```bash
# Normal run - only loads new/changed schemas
python main.py

# Force reload all schemas
python main.py --force-reload
```

## Schema JSON Format

Each table should have a JSON file in `schemas/`. Two formats are supported:

### Format 1: Columns as Dictionary (Recommended)

```json
{
  "table_name": "customers",
  "description": "Brief table description",
  "business_context": "How this table is used in business logic",
  "columns": {
    "id": {
      "type": "INTEGER",
      "constraints": "PRIMARY KEY, AUTO_INCREMENT",
      "description": "Column purpose"
    },
    "name": {
      "type": "VARCHAR(255)",
      "constraints": "NOT NULL",
      "description": "Customer full name"
    }
  },
  "relationships": [
    {
      "type": "one_to_many",
      "related_table": "orders",
      "foreign_key": "customer_id",
      "description": "Relationship explanation"
    }
  ]
}
```

### Format 2: Columns as List (Also Supported)

```json
{
  "table_name": "reviews",
  "description": "Product reviews and ratings",
  "business_context": "Customer feedback on products",
  "columns": [
    {
      "name": "id",
      "type": "INTEGER",
      "constraints": ["PRIMARY KEY", "AUTO_INCREMENT"],
      "description": "Unique review identifier"
    },
    {
      "name": "rating",
      "type": "INTEGER",
      "constraints": ["NOT NULL", "CHECK (rating >= 1 AND rating <= 5)"],
      "description": "Star rating from 1 to 5"
    }
  ],
  "relationships": [
    "Each review belongs to one product",
    "Each review is written by one customer"
  ]
}
```

> **Note**: Relationships can be an array of objects or an array of strings.

## Usage

### Basic Usage

```python
from agent.llm_provider import create_llm_provider
from agent.rag_manager import SchemaRAGManager
from agent.sql_agent import SQLAgent

# Initialize components (uses settings.py)
llm = create_llm_provider("openai", model="gpt-4")
rag = SchemaRAGManager()  # Automatically uses settings

# Load schemas (intelligently detects changes)
rag.load_schemas_from_directory()  # Only loads new/changed files

# Or force reload everything
# rag.load_schemas_from_directory(force_reload=True)

# Create agent
agent = SQLAgent(llm_provider=llm, rag_manager=rag)

# Generate SQL
query = agent.generate_query("Get all active customers from last month")
print(query)
```

### Switch Models

```python
# OpenAI
llm = create_llm_provider("openai", model="gpt-4")

# Anthropic Claude
llm = create_llm_provider("anthropic", model="claude-3-5-sonnet-20241022")

# Local Ollama
llm = create_llm_provider("ollama", model="llama2")
```

### Interactive Mode

```python
agent.interactive_mode()
```

## Project Structure

```
genai-training/
├── schemas/              # Table JSON files
│   ├── customers.json
│   ├── orders.json
│   ├── order_items.json
│   └── products.json
├── agent/
│   ├── __init__.py
│   ├── llm_provider.py   # Model abstraction
│   ├── rag_manager.py    # ChromaDB operations
│   └── sql_agent.py      # Main agent
├── examples/
│   └── example_schemas.py
├── chroma_db/            # ChromaDB persistent storage
├── main.py               # Entry point
├── requirements.txt
└── .env                  # API keys
```

## Configuration

**Settings** (`settings.py`):

```python
CHROMA_DIRECTORY = "chroma_db"
CHROMA_COLLECTION_NAME = "database_schemas"
DATABASE_SCHEMAS_DIR = "schemas"
```

**LLM Configuration** (`main.py`):

```python
LLM_PROVIDER = "openai"  # or "anthropic", "ollama"
MODEL = "gpt-4"          # Model name
```

## How It Works

1. **Smart Schema Loading**:
   - Calculates MD5 hash for each JSON file
   - Compares with stored hash in ChromaDB metadata
   - Only loads new or modified files (skips unchanged)
   - Automatically removes deleted schemas from ChromaDB
2. **Query Analysis**: User's business logic analyzed semantically

3. **Retrieval**: Top-K most relevant tables retrieved (not all 100!)

4. **Generation**: LLM generates SQL with only relevant context

5. **Result**: Clean SQL query returned

## Benefits of RAG Approach

For 100 tables:

- **Token Savings**: 95% reduction (2K vs 50K tokens)
- **Cost Savings**: ~30x cheaper per query
- **Speed**: 3-5x faster responses
- **Accuracy**: Better focus on relevant tables
- **Efficiency**: Only loads changed schemas (not all 100 every time)

## Schema Change Management

### Adding New Tables

Simply create a new JSON file in `schemas/`:

```bash
# Create new schema
cat > schemas/payments.json << EOF
{
  "table_name": "payments",
  "description": "Payment transactions",
  ...
}
EOF

# Run agent - it will automatically detect and load new schemas
python main.py
```

Output: `✨ New: payments.json`

### Editing Existing Tables

Edit any schema JSON file and run the agent:

```bash
# Edit a schema file
vim schemas/customers.json

# Run agent - only modified schemas will reload
python main.py
```

Output: `🔄 Updated: customers.json`

### Deleting Tables

Remove a schema file and run the agent:

```bash
# Delete a schema
rm schemas/old_table.json

# Run agent - it will automatically clean up ChromaDB
python main.py
```

Output: `🗑️ Removed: old_table (file no longer exists)`

### Status Indicators

- ✨ **New**: First time loading this schema
- 🔄 **Updated**: File changed since last load
- ⏭️ **Skipped**: No changes detected (fast!)
- 🗑️ **Removed**: File no longer exists

### Force Reload All Schemas

If you need to reload everything regardless of changes:

```bash
python main.py --force-reload
# or
python main.py -f
```

## License

MIT
