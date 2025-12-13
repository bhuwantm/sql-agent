"""Main script to run the SQL agent."""

import settings

from agent.llm_provider import create_llm_provider
from agent.rag_manager import SchemaRAGManager
from agent.sql_agent import SQLAgent


def setup_rag_manager(force_reload: bool = False) -> SchemaRAGManager:
    """
    Set up and return configured RAG manager with ChromaDB.
    Uses settings from settings.py.
    
    Args:
        force_reload: If True, reload all schemas regardless of changes
        
    Returns:
        Configured SchemaRAGManager instance
    """
    rag_manager = SchemaRAGManager()
    
    # Load schemas - intelligently detects new/changed files
    if rag_manager.collection.count() == 0:
        print(f"\n📁 Loading schemas from '{settings.DATABASE_SCHEMAS_DIR}/'...")
        rag_manager.load_schemas_from_directory(force_reload=False)
    else:
        print(f"\n🔄 Checking for schema updates in '{settings.DATABASE_SCHEMAS_DIR}/'...")
        # Remove any deleted schemas
        removed = rag_manager.remove_deleted_schemas()
        if removed > 0:
            print(f"Cleaned up {removed} deleted schema(s)")
        
        # Load new or modified schemas
        rag_manager.load_schemas_from_directory(force_reload=force_reload)
    
    return rag_manager


def setup_llm_provider(provider: str = "bedrock", model: str = None):
    """
    Set up and return configured LLM provider.
    
    Args:
        provider: LLM provider ("openai", "anthropic", "ollama")
        model: Model name (provider-specific)
        
    Returns:
        Configured LLM provider instance
    """
    llm_kwargs = {}
    if model:
        llm_kwargs['model'] = model
    
    print(f"\n🤖 Initializing {provider.upper()} LLM...")
    llm_provider = create_llm_provider(provider, **llm_kwargs)
    
    return llm_provider


def setup_agent(
    provider: str = "bedrock",
    model: str = None,
    top_k_schemas: int = 5
) -> SQLAgent:
    """
    Set up and return configured SQL agent.
    
    Args:
        provider: LLM provider ("openai", "anthropic", "ollama")
        model: Model name (provider-specific)
        top_k_schemas: Number of relevant schemas to retrieve
        
    Returns:
        Configured SQLAgent instance
    """
  
    rag_manager = setup_rag_manager()
    llm_provider = setup_llm_provider(provider=provider, model=model)
    
    # Create agent
    agent = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        top_k_schemas=top_k_schemas
    )
    
    print("✓ Agent ready!\n")
    return agent


def main(force_reload: bool = False):
    """
    Main function to demonstrate the SQL agent.
    
    Args:
        force_reload: If True, force reload all schemas from disk
    """
    
    # Configuration (environment variables loaded via settings.py)
    LLM_PROVIDER = "bedrock"  # Options: "openai", "anthropic", "ollama", "bedrock"
    MODEL = settings.BEDROCK_MODEL  # or "gpt-4", "claude-3-5-sonnet-20241022", etc.
    
    # Step 1: Setup ChromaDB and populate embeddings
    print("\n" + "="*60)
    print("STEP 1: Setting up ChromaDB RAG")
    print("="*60)
    rag_manager = setup_rag_manager(force_reload=force_reload)
    print("✓ ChromaDB ready with embeddings!\n")
    
    # Step 2: Setup LLM provider
    print("="*60)
    print("STEP 2: Setting up LLM Provider")
    print("="*60)
    llm_provider = setup_llm_provider(provider=LLM_PROVIDER, model=MODEL)
    print("✓ LLM ready!\n")
    
    # Step 3: Create agent
    print("="*60)
    print("STEP 3: Creating SQL Agent")
    print("="*60)
    agent = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        top_k_schemas=5
    )
    print("✓ Agent ready!\n")
    
    # Example queries
    print("=" * 50)
    print("Example Queries")
    print("=" * 50)
    
    examples = [
        "Get all active customers who registered in the last 30 days",
        "Show total revenue by customer for orders placed this year",
        "Find customers who have placed more than 5 orders",
    ]
    
    for i, business_logic in enumerate(examples, 1):
        print(f"\n{i}. Business Logic: {business_logic}")
        print("-" * 50)
        
        # query = agent.generate_query(business_logic)
        # print(f"Generated SQL:\n{query}\n")
    
    # Interactive mode (optional)
    print("\n" + "=" * 50)
    run_interactive = input("Run interactive mode? (y/n): ").strip().lower()
    if run_interactive == 'y':
        agent.interactive_mode(explain=False)


if __name__ == "__main__":
    import sys
    
    # Check for --force-reload flag
    force_reload = "--force-reload" in sys.argv or "-f" in sys.argv
    
    if force_reload:
        print("🔄 Force reload enabled - all schemas will be reloaded\n")
    
    main(force_reload=force_reload)
