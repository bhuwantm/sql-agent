"""Example demonstrating conversation history feature."""

import settings
from llm_providers import create_llm_provider
from rag import SchemaRAGManager
from agent.sql_agent import SQLAgent


def demo_conversation_history():
    """Demonstrate conversation history in action."""
    
    print("=" * 70)
    print("CONVERSATION HISTORY DEMO")
    print("=" * 70)
    
    # Setup
    rag_manager = SchemaRAGManager()
    llm_provider = create_llm_provider("bedrock", model=settings.BEDROCK_MODEL)
    
    # Create agent WITH conversation history (token-optimized)
    agent_with_history = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        enable_conversation_history=True,
        max_history_length=10,      # Store 10 turns
        history_in_prompt=3,        # Send only last 3 (saves tokens)
        summarize_old_turns=True    # Truncate long SQL responses
    )
    
    print("\n📚 Example: Follow-up questions with context\n")
    
    # First query
    print("1️⃣  User: 'Get all customers'")
    query1 = agent_with_history.generate_query("Get all customers")
    print(f"   Agent: {query1[:80]}...\n")
    
    # Follow-up query (refers to previous context)
    print("2️⃣  User: 'Only show active ones'")
    query2 = agent_with_history.generate_query("Only show active ones")
    print(f"   Agent: {query2[:80]}...\n")
    
    # Another follow-up
    print("3️⃣  User: 'Add their email addresses'")
    query3 = agent_with_history.generate_query("Add their email addresses")
    print(f"   Agent: {query3[:80]}...\n")
    
    # Show history
    print("\n📜 Conversation History:")
    print("-" * 70)
    for i, turn in enumerate(agent_with_history.get_history(), 1):
        print(f"{i}. User: {turn['user']}")
        print(f"   Response: {turn['assistant'][:60]}...\n")
    
    # Clear and start fresh
    agent_with_history.clear_history()
    print("✓ History cleared - agent ready for new conversation\n")
    
    print("=" * 70)
    print("💡 TIP: In interactive mode, use 'history' and 'clear' commands")
    print("=" * 70)


if __name__ == "__main__":
    demo_conversation_history()
