"""Utility script to inspect ChromaDB contents."""

from rag_manager import SchemaRAGManager
import settings
import json


def view_chroma_contents():
    """View all contents stored in ChromaDB."""
    
    # Connect to existing ChromaDB
    rag = SchemaRAGManager()
    
    print("\n" + "="*60)
    print("CHROMADB COLLECTION INSPECTOR")
    print("="*60)
    
    # 1. Collection stats
    count = rag.collection.count()
    print(f"\n📊 Total schemas stored: {count}")
    
    # 2. List all table names
    table_names = rag.get_all_table_names()
    print(f"\n📋 Table names: {', '.join(table_names)}")
    
    # 3. Get all data
    print("\n" + "="*60)
    print("DETAILED CONTENTS")
    print("="*60)
    
    results = rag.collection.get(
        include=['documents', 'metadatas', 'embeddings']
    )
    
    # Display each schema
    for i, (table_id, document, metadata) in enumerate(zip(
        results['ids'],
        results['documents'],
        results['metadatas']
    ), 1):
        print(f"\n{'─'*60}")
        print(f"[{i}] Table: {table_id}")
        print(f"{'─'*60}")
        
        # Searchable text (what gets embedded)
        print("\n📝 Searchable Text (Document):")
        print(document)
        
        # Full schema JSON (just show table name and description, not full JSON)
        print("\n🗂️  Schema Summary:")
        print(f"  Table: {metadata.get('table_name', 'N/A')}")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        print(f"  Business Context: {metadata.get('business_context', 'N/A')}")
        
        # Embedding vector (first 10 dimensions only)
        if results.get('embeddings') is not None and len(results['embeddings']) > i-1:
            embedding = results['embeddings'][i-1]
            if embedding is not None and len(embedding) > 0:
                print(f"\n🔢 Embedding Vector:")
                print(f"   First 10 dimensions: {embedding[:10]}")
                print(f"   Total dimensions: {len(embedding)}")
                print(f"   Vector magnitude: {sum(x**2 for x in embedding)**0.5:.4f}")
    
    print("\n" + "="*60)
    
    # 4. Test a query (if collection is not empty)
    if count > 0:
        print("\n🔍 TEST QUERY")
        print("="*60)
        test_query = "customer information and orders"
        print(f"Query: '{test_query}'")
        
        relevant = rag.retrieve_relevant_schemas(test_query, top_k=3)
        print(f"\nTop {len(relevant)} relevant tables:")
        for schema in relevant:
            print(f"  - {schema['table_name']}: {schema.get('description', 'N/A')}")
    else:
        print("\n⚠️  ChromaDB collection is empty!")
        print(f"Run 'python main.py' first to load schemas from '{settings.DATABASE_SCHEMAS_DIR}/'.")


def search_by_query():
    """Interactive search through ChromaDB."""
    
    rag = SchemaRAGManager()
    
    print("\n" + "="*60)
    print("INTERACTIVE CHROMADB SEARCH")
    print("="*60)
    print(f"Available tables: {', '.join(rag.get_all_table_names())}")
    print("\nType 'exit' to quit\n")
    
    while True:
        query = input("🔍 Enter search query: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\n📊 Searching for: '{query}'")
        relevant = rag.retrieve_relevant_schemas(query, top_k=5)
        
        print(f"\n✓ Found {len(relevant)} relevant schemas:\n")
        for i, schema in enumerate(relevant, 1):
            print(f"{i}. {schema['table_name']}")
            print(f"   Description: {schema.get('description', 'N/A')}")
            print(f"   Columns: {', '.join(schema.get('columns', {}).keys())}")
            print()


def view_single_schema(table_name: str):
    """View a specific schema by table name."""
    
    rag = SchemaRAGManager()
    
    schema = rag.get_schema_by_name(table_name)
    
    if schema:
        print(f"\n📋 Schema for '{table_name}':")
        print(json.dumps(schema, indent=2))
    else:
        print(f"\n❌ Table '{table_name}' not found")
        print(f"Available tables: {', '.join(rag.get_all_table_names())}")


def view_embeddings(table_name: str = None):
    """View embeddings for a specific table or all tables."""
    
    rag = SchemaRAGManager()
    
    if table_name:
        # View embeddings for specific table
        try:
            results = rag.collection.get(ids=[table_name], include=['embeddings'])
            
            if not results['ids']:
                print(f"\n❌ Table '{table_name}' not found")
                print(f"Available tables: {', '.join(rag.get_all_table_names())}")
                return
            
            embedding = results['embeddings'][0]
            
            print(f"\n🔢 Embeddings for '{table_name}':")
            print("=" * 60)
            print(f"Dimensions: {len(embedding)}")
            print(f"Magnitude: {sum(x**2 for x in embedding)**0.5:.4f}")
            print(f"\nFirst 20 values:")
            print(embedding[:20])
            print(f"\nLast 20 values:")
            print(embedding[-20:])
            print(f"\nFull vector ({len(embedding)} dimensions):")
            print(embedding)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        # View embeddings summary for all tables
        results = rag.collection.get(include=['embeddings'])
        
        print("\n" + "="*60)
        print("EMBEDDINGS SUMMARY")
        print("="*60)
        
        for i, (table_id, embedding) in enumerate(zip(results['ids'], results['embeddings']), 1):
            print(f"\n{i}. Table: {table_id}")
            print(f"   Dimensions: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            print(f"   Magnitude: {sum(x**2 for x in embedding)**0.5:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "search":
            search_by_query()
        elif command == "view" and len(sys.argv) > 2:
            view_single_schema(sys.argv[2])
        elif command == "embeddings":
            if len(sys.argv) > 2:
                view_embeddings(sys.argv[2])
            else:
                view_embeddings()
        else:
            print("Usage:")
            print("  python view_chroma.py                   # View all contents")
            print("  python view_chroma.py search            # Interactive search")
            print("  python view_chroma.py view <table>      # View specific table schema")
            print("  python view_chroma.py embeddings        # View embeddings summary for all tables")
            print("  python view_chroma.py embeddings <table> # View embeddings for specific table")
    else:
        view_chroma_contents()
