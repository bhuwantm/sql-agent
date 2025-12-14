"""SQL generation agent using RAG and LLM."""

from typing import Dict, Any, List
from rag import SchemaRAGManager
from llm_providers import LLMProvider
from .sql_agent_prompts import create_sql_prompt


class SQLAgent:
    """Agent that generates SQL queries from business logic."""
    
    def __init__(
        self, 
        llm_provider: LLMProvider,
        rag_manager: SchemaRAGManager,
        top_k_schemas: int = 5,
        enable_conversation_history: bool = True,
        max_history_length: int = 10,
        history_in_prompt: int = 3,
        summarize_old_turns: bool = True
    ):
        """
        Initialize SQL agent.
        
        Args:
            llm_provider: LLM provider instance
            rag_manager: RAG manager with loaded schemas
            top_k_schemas: Number of relevant schemas to retrieve
            enable_conversation_history: Whether to maintain conversation context
            max_history_length: Maximum number of conversation turns to remember (stored)
            history_in_prompt: Number of recent turns to include in prompts (to reduce tokens)
            summarize_old_turns: If True, only send user questions for old turns, not full SQL responses
        """
        self.llm = llm_provider
        self.rag_manager = rag_manager
        self.top_k = top_k_schemas
        self.enable_conversation_history = enable_conversation_history
        self.max_history_length = max_history_length
        self.history_in_prompt = history_in_prompt
        self.summarize_old_turns = summarize_old_turns
        self.conversation_history: List[Dict[str, str]] = []
    
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
        relevant_schemas = self.rag_manager.retrieve_relevant_schemas(
            business_logic, 
            top_k=self.top_k
        )
        
        if not relevant_schemas:
            return "Error: No relevant tables found for this query."
        
        # Format schemas for prompt
        schema_context = self._format_schemas(relevant_schemas)
        
        # Prepare conversation history for prompt (optimized for tokens)
        history_for_prompt = None
        if self.enable_conversation_history and self.conversation_history:
            history_for_prompt = self._prepare_history_for_prompt()
        
        # Create prompt using centralized prompt builder (with conversation history)
        prompt = create_sql_prompt(
            business_logic, 
            schema_context, 
            explain,
            conversation_history=history_for_prompt
        )
        
        # Generate SQL
        response = self.llm.generate(prompt, temperature=0.0)
        
        # Store in conversation history
        if self.enable_conversation_history:
            self._add_to_history(business_logic, response.strip())
        
        return response.strip()
    
    def _prepare_history_for_prompt(self) -> List[Dict[str, str]]:
        """
        Prepare conversation history for inclusion in prompts.
        Optimizes token usage by:
        1. Only including last N turns (history_in_prompt)
        2. Summarizing older turns (keeping only user questions)
        
        Returns:
            List of conversation turns optimized for prompt inclusion
        """
        if not self.conversation_history:
            return []
        
        # Get recent turns based on history_in_prompt setting
        recent_turns = self.conversation_history[-self.history_in_prompt:]
        
        if not self.summarize_old_turns:
            # Return full turns as-is
            return recent_turns
        
        # Summarize: keep only user questions, truncate long SQL responses
        optimized_turns = []
        for turn in recent_turns:
            user_msg = turn['user']
            assistant_msg = turn['assistant']
            
            # For SQL responses longer than 200 chars, just indicate a query was generated
            if len(assistant_msg) > 200:
                assistant_summary = f"[Generated SQL query: {assistant_msg[:100]}...]"
                optimized_turns.append({
                    'user': user_msg,
                    'assistant': assistant_summary
                })
            else:
                optimized_turns.append(turn)
        
        return optimized_turns
    
    def _prepare_history_for_prompt(self) -> List[Dict[str, str]]:
        """
        Prepare conversation history for inclusion in prompts.
        Optimizes token usage by:
        1. Only including last N turns (history_in_prompt)
        2. Summarizing older turns (keeping only user questions + truncated responses)
        
        Returns:
            List of conversation turns optimized for prompt inclusion
        """
        if not self.conversation_history:
            return []
        
        # Get recent turns based on history_in_prompt setting
        recent_turns = self.conversation_history[-self.history_in_prompt:]
        
        if not self.summarize_old_turns:
            # Return full turns as-is
            return recent_turns
        
        # Summarize: truncate long SQL responses to save tokens
        optimized_turns = []
        for turn in recent_turns:
            user_msg = turn['user']
            assistant_msg = turn['assistant']
            
            # For SQL responses longer than 200 chars, truncate them
            if len(assistant_msg) > 200:
                assistant_summary = f"[Generated SQL query: {assistant_msg[:100]}...]"
                optimized_turns.append({
                    'user': user_msg,
                    'assistant': assistant_summary
                })
            else:
                optimized_turns.append(turn)
        
        return optimized_turns
    
    def _add_to_history(self, user_input: str, assistant_response: str) -> None:
        """Add a turn to conversation history."""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()
    
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
        print("Available tables:", ", ".join(self.rag_manager.get_all_table_names()))
        if self.enable_conversation_history:
            print("💬 Conversation history: ENABLED")
            print("   Commands: 'clear' (reset history), 'history' (view history)")
        print("\nType 'exit' or 'quit' to stop\n")
        
        while True:
            try:
                user_input = input("📝 Business Logic: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == 'clear':
                    self.clear_history()
                    print("✓ Conversation history cleared\n")
                    continue
                
                if user_input.lower() == 'history':
                    self._show_history()
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
    
    def _show_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            print("📭 No conversation history yet\n")
            return
        
        print("\n📜 Conversation History:")
        print("=" * 50)
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"\n{i}. User: {turn['user']}")
            print(f"   Assistant: {turn['assistant'][:100]}...")  # Truncate long responses
        print("\n" + "=" * 50 + "\n")
