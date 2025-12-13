"""SQL generation agent using RAG and LLM."""

from typing import Dict, Any, List
from .rag_manager import SchemaRAGManager
from .llm_provider import LLMProvider
from .prompts import create_sql_prompt


class SQLAgent:
    """Agent that generates SQL queries from business logic."""
    
    def __init__(
        self, 
        llm_provider: LLMProvider,
        rag_manager: SchemaRAGManager,
        top_k_schemas: int = 5
    ):
        """
        Initialize SQL agent.
        
        Args:
            llm_provider: LLM provider instance
            rag_manager: RAG manager with loaded schemas
            top_k_schemas: Number of relevant schemas to retrieve
        """
        self.llm = llm_provider
        self.rag = rag_manager
        self.top_k = top_k_schemas
    
    def generate_query(
        self, 
        business_logic: str,
        explain: bool = False
    ) -> str:
        """
        Generate SQL query from business logic.
        
        Args:
            business_logic: Natural language description of what to query
            explain: Whether to include explanation with the query
            
        Returns:
            SQL query string
        """
        # Retrieve relevant schemas
        relevant_schemas = self.rag.retrieve_relevant_schemas(
            business_logic, 
            top_k=self.top_k
        )
        
        if not relevant_schemas:
            return "Error: No relevant tables found for this query."
        
        # Format schemas for prompt
        schema_context = self._format_schemas(relevant_schemas)
        
        # Create prompt using centralized prompt builder
        prompt = create_sql_prompt(business_logic, schema_context, explain)
        
        # Generate SQL
        response = self.llm.generate(prompt, temperature=0.0)
        
        return response.strip()
    
    def _format_schemas(self, schemas: List[Dict[str, Any]]) -> str:
        """Format schemas into readable text for prompt."""
        formatted = []
        
        for schema in schemas:
            table_info = [f"\n### Table: {schema['table_name']}"]
            
            if schema.get('description'):
                table_info.append(f"Description: {schema['description']}")
            
            # Columns (handle both dict and list formats)
            columns = schema.get('columns', {})
            if columns:
                table_info.append("\nColumns:")
                
                # Handle columns as dictionary
                if isinstance(columns, dict):
                    for col_name, col_info in columns.items():
                        col_line = f"  - {col_name} ({col_info.get('type', 'UNKNOWN')})"
                        if col_info.get('constraints'):
                            col_line += f" [{col_info['constraints']}]"
                        if col_info.get('description'):
                            col_line += f" - {col_info['description']}"
                        table_info.append(col_line)
                
                # Handle columns as list
                elif isinstance(columns, list):
                    for col_info in columns:
                        col_name = col_info.get('name', 'UNKNOWN')
                        col_type = col_info.get('type', 'UNKNOWN')
                        col_line = f"  - {col_name} ({col_type})"
                        if col_info.get('constraints'):
                            constraints = col_info['constraints']
                            if isinstance(constraints, list):
                                constraints = ', '.join(constraints)
                            col_line += f" [{constraints}]"
                        if col_info.get('description'):
                            col_line += f" - {col_info['description']}"
                        table_info.append(col_line)
            
            # Relationships (handle both string and dict formats)
            if schema.get('relationships'):
                table_info.append("\nRelationships:")
                for rel in schema['relationships']:
                    if isinstance(rel, str):
                        table_info.append(f"  - {rel}")
                    elif isinstance(rel, dict):
                        table_info.append(f"  - {rel.get('description', str(rel))}")
            
            formatted.append("\n".join(table_info))
        
        return "\n".join(formatted)
    
    def interactive_mode(self, explain: bool) -> None:
        """Run agent in interactive mode."""
        print("\n🤖 SQL Agent Interactive Mode")
        print("=" * 50)
        print("Available tables:", ", ".join(self.rag.get_all_table_names()))
        print("\nType 'exit' or 'quit' to stop\n")
        
        while True:
            try:
                user_input = input("📝 Business Logic: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🔍 Generating SQL query...\n")
                query = self.generate_query(business_logic=user_input, explain=explain)
                print(f"✓ Generated Query:\n{query}\n")
                print("-" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
