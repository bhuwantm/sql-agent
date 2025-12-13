"""ChromaDB RAG manager for database schemas."""

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*", category=Warning)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import chromadb
from typing import List, Dict, Any, Optional

from settings import CHROMA_DIRECTORY, CHROMA_COLLECTION_NAME, DATABASE_SCHEMAS_DIR


class SchemaRAGManager:
    """Manages database schemas in ChromaDB for retrieval."""
    
    def __init__(self):
        """
        Initialize RAG manager.
        """

        self.client = chromadb.PersistentClient(path=CHROMA_DIRECTORY)
    
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(CHROMA_COLLECTION_NAME)
            print(
                f"Loaded existing collection '{CHROMA_COLLECTION_NAME}' with {self.collection.count()} schemas"
            )
        except:
            self.collection = self.client.create_collection(
                CHROMA_COLLECTION_NAME,
                metadata={"description": "Database table schemas for SQL generation"}
            )
            print(f"Created new collection '{CHROMA_COLLECTION_NAME}'")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content for change detection."""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_stored_metadata(self, table_name: str) -> dict:
        """Get stored metadata including file hash."""
        try:
            result = self.collection.get(ids=[table_name], include=['metadatas'])
            if result['metadatas'] and result['metadatas'][0]:
                return result['metadatas'][0]
        except:
            pass
        return {}
    
    def load_schemas_from_directory(self, force_reload: bool = False) -> dict:
        """
        Load JSON schema files from directory into ChromaDB.
        Only loads new or modified files unless force_reload=True.
        
        Args:
            force_reload: If True, reload all schemas even if unchanged
            
        Returns:
            Dictionary with counts: {'new': int, 'updated': int, 'skipped': int, 'errors': int}
        """
        schemas_path = Path(DATABASE_SCHEMAS_DIR)
        if not schemas_path.exists():
            raise FileNotFoundError(f"Schemas directory not found: {DATABASE_SCHEMAS_DIR}")
        
        json_files = list(schemas_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {DATABASE_SCHEMAS_DIR}")
        
        # Track existing tables in ChromaDB
        existing_tables = set(self.get_all_table_names())
        
        counts = {'new': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
        
        for json_file in json_files:
            try:
                # Calculate file hash
                file_hash = self._get_file_hash(json_file)
                
                # Load schema
                with open(json_file, 'r') as f:
                    schema = json.load(f)
                
                table_name = schema.get('table_name')
                if not table_name:
                    print(f"⚠️  Skipping {json_file.name}: no table_name field")
                    counts['errors'] += 1
                    continue
                
                # Check if schema exists and if it's changed
                is_new = table_name not in existing_tables
                
                if not force_reload and not is_new:
                    # Check if file has changed
                    stored_metadata = self._get_stored_metadata(table_name)
                    stored_hash = stored_metadata.get('file_hash', '')
                    
                    if stored_hash == file_hash:
                        print(f"⏭️  Skipped: {json_file.name} (unchanged)")
                        counts['skipped'] += 1
                        continue
                
                # Add/update schema with file hash
                self.add_schema(schema, file_hash=file_hash)
                
                if is_new:
                    print(f"✨ New: {json_file.name}")
                    counts['new'] += 1
                else:
                    print(f"🔄 Updated: {json_file.name}")
                    counts['updated'] += 1
                    
            except Exception as e:
                print(f"❌ Error loading {json_file.name}: {e}")
                counts['errors'] += 1
        
        # Summary
        total = counts['new'] + counts['updated'] + counts['skipped']
        print(f"\n✓ Summary: {counts['new']} new, {counts['updated']} updated, {counts['skipped']} unchanged")
        if counts['errors'] > 0:
            print(f"⚠️  {counts['errors']} errors")
        
        return counts
    
    def add_schema(self, schema: Dict[str, Any], file_hash: str = None) -> None:
        """
        Add or update a single table schema in ChromaDB.
        
        Args:
            schema: Dictionary containing table schema information
            file_hash: MD5 hash of the source file for change tracking
        """
        table_name = schema.get("table_name")
        if not table_name:
            raise ValueError("Schema must contain 'table_name' field")
        
        # Create searchable text representation
        searchable_text = self._create_searchable_text(schema)
        
        # Flatten metadata - ChromaDB only accepts str, int, float, bool
        # Store the full schema as JSON string
        flattened_metadata = {
            "table_name": schema.get("table_name"),
            "description": schema.get("description", ""),
            "business_context": schema.get("business_context", ""),
            "schema_json": json.dumps(schema),  # Store full schema as JSON string
            "file_hash": file_hash or ""  # Track file hash for change detection
        }
        
        # Add to ChromaDB (upsert will update if exists, create if new)
        self.collection.upsert(
            ids=[table_name],
            documents=[searchable_text],
            metadatas=[flattened_metadata]
        )
    
    def _create_searchable_text(self, schema: Dict[str, Any]) -> str:
        """Create rich searchable text from schema."""
        parts = []
        
        # Table name and description
        parts.append(f"Table: {schema['table_name']}")
        if schema.get('description'):
            parts.append(f"Description: {schema['description']}")
        if schema.get('business_context'):
            parts.append(f"Business Context: {schema['business_context']}")
        
        # Columns with descriptions (handle both dict and list formats)
        if 'columns' in schema:
            parts.append("\nColumns:")
            columns = schema['columns']
            
            # Handle columns as dictionary (key: column_name, value: column_info)
            if isinstance(columns, dict):
                for col_name, col_info in columns.items():
                    col_desc = col_info.get('description', '')
                    parts.append(f"  - {col_name}: {col_info.get('type', '')} {col_desc}")
            
            # Handle columns as list (each item has 'name' key)
            elif isinstance(columns, list):
                for col_info in columns:
                    col_name = col_info.get('name', '')
                    col_type = col_info.get('type', '')
                    col_desc = col_info.get('description', '')
                    parts.append(f"  - {col_name}: {col_type} {col_desc}")
        
        # Relationships (handle both string list and object list formats)
        if schema.get('relationships'):
            parts.append("\nRelationships:")
            for rel in schema['relationships']:
                if isinstance(rel, str):
                    parts.append(f"  - {rel}")
                elif isinstance(rel, dict):
                    parts.append(f"  - {rel.get('description', rel.get('type', ''))}")
        
        return "\n".join(parts)
    
    def retrieve_relevant_schemas(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant schemas for a query.
        
        Args:
            query: Natural language query or business logic
            top_k: Number of schemas to retrieve
            
        Returns:
            List of schema dictionaries
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        
        if not results['metadatas'] or not results['metadatas'][0]:
            return []
        
        # Reconstruct full schema from JSON string
        schemas = []
        for metadata in results['metadatas'][0]:
            if 'schema_json' in metadata:
                schema = json.loads(metadata['schema_json'])
                schemas.append(schema)
        
        return schemas
    
    def get_all_table_names(self) -> List[str]:
        """Get list of all table names in the collection."""
        results = self.collection.get()
        return results['ids'] if results['ids'] else []
    
    def get_schema_by_name(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get specific schema by table name."""
        try:
            result = self.collection.get(ids=[table_name])
            if result['metadatas'] and result['metadatas'][0]:
                metadata = result['metadatas'][0]
                if 'schema_json' in metadata:
                    return json.loads(metadata['schema_json'])
                return metadata
        except:
            pass
        return None
    
    def clear_collection(self) -> None:
        """Clear all schemas from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)
    
    def remove_deleted_schemas(self) -> int:
        """
        Remove schemas from ChromaDB that no longer have corresponding JSON files.
        
        Returns:
            Number of schemas removed
        """
        schemas_path = Path(DATABASE_SCHEMAS_DIR)
        if not schemas_path.exists():
            return 0
        
        # Get all JSON files in directory
        json_files = {f.stem for f in schemas_path.glob("*.json")}
        
        # Get all table names in ChromaDB
        stored_tables = set(self.get_all_table_names())
        
        # Find orphaned schemas (in DB but not in files)
        to_remove = stored_tables - json_files
        
        if to_remove:
            self.collection.delete(ids=list(to_remove))
            for table in to_remove:
                print(f"🗑️  Removed: {table} (file no longer exists)")
        
        return len(to_remove)
        print("Collection cleared")
