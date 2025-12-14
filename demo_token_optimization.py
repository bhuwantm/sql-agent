"""Demonstration of token optimization strategies for conversation history."""

import settings
from llm_providers import create_llm_provider
from rag import SchemaRAGManager
from agent.sql_agent import SQLAgent


def demo_token_optimization():
    """Show different token optimization configurations."""
    
    print("=" * 80)
    print("TOKEN OPTIMIZATION STRATEGIES FOR CONVERSATION HISTORY")
    print("=" * 80)
    
    # Setup
    rag_manager = SchemaRAGManager()
    llm_provider = create_llm_provider("bedrock", model=settings.BEDROCK_MODEL)
    
    print("\n📊 STRATEGY COMPARISON:\n")
    
    # Strategy 1: No History (Baseline - Cheapest)
    print("1️⃣  NO HISTORY (Baseline)")
    print("   - Cost: Lowest (no extra tokens)")
    print("   - Context: None between queries")
    print("   - Use case: Independent queries, cost-sensitive")
    print()
    
    agent_no_history = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        enable_conversation_history=False
    )
    
    # Strategy 2: Full History (Most Expensive)
    print("2️⃣  FULL HISTORY (Most Expensive)")
    print("   - Store: 10 turns")
    print("   - Send: All 10 turns with full SQL queries")
    print("   - Cost: Highest (can be 1000+ tokens per turn)")
    print("   - Use case: Maximum context retention")
    print()
    
    agent_full = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        enable_conversation_history=True,
        max_history_length=10,
        history_in_prompt=10,
        summarize_old_turns=False
    )
    
    # Strategy 3: Recent + Summarized (Recommended)
    print("3️⃣  RECENT + SUMMARIZED (Recommended ✨)")
    print("   - Store: 10 turns")
    print("   - Send: Last 3 turns only")
    print("   - SQL responses truncated to 100 chars")
    print("   - Cost: ~60% reduction vs full history")
    print("   - Use case: Balanced context + cost")
    print()
    
    agent_optimized = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        enable_conversation_history=True,
        max_history_length=10,
        history_in_prompt=3,
        summarize_old_turns=True
    )
    
    # Strategy 4: Minimal (Ultra-Cheap)
    print("4️⃣  MINIMAL CONTEXT (Ultra-Cheap)")
    print("   - Store: 5 turns")
    print("   - Send: Last 1 turn only, summarized")
    print("   - Cost: ~80% reduction vs full history")
    print("   - Use case: Simple follow-ups only")
    print()
    
    agent_minimal = SQLAgent(
        llm_provider=llm_provider,
        rag_manager=rag_manager,
        enable_conversation_history=True,
        max_history_length=5,
        history_in_prompt=1,
        summarize_old_turns=True
    )
    
    print("=" * 80)
    print("💡 TOKEN USAGE ESTIMATES (per query with 5 turn history):")
    print("=" * 80)
    print()
    print("Strategy                    | Tokens/Query | Monthly Cost* | Best For")
    print("-" * 80)
    print("No History                  |     500      |    $15       | Independent queries")
    print("Full History (10 turns)     |    3,500     |   $105       | Research/analysis")
    print("Recent + Summarized (3)     |    1,200     |    $36       | Most use cases ✨")
    print("Minimal (1 turn)            |     700      |    $21       | Simple refinements")
    print()
    print("*Based on 1000 queries/month at $0.003/1K input tokens (Nova Lite pricing)")
    print()
    
    print("=" * 80)
    print("🎯 RECOMMENDATIONS:")
    print("=" * 80)
    print()
    print("✅ Start with: history_in_prompt=3, summarize_old_turns=True")
    print("✅ For complex workflows: history_in_prompt=5")
    print("✅ For cost optimization: history_in_prompt=1 or disable history")
    print("✅ Monitor token usage in production with LLM provider dashboards")
    print()
    
    print("=" * 80)
    print("📝 CONFIGURATION EXAMPLE:")
    print("=" * 80)
    print()
    print("agent = SQLAgent(")
    print("    llm_provider=llm_provider,")
    print("    rag_manager=rag_manager,")
    print("    enable_conversation_history=True,")
    print("    max_history_length=10,          # Store 10 turns")
    print("    history_in_prompt=3,            # Send only last 3 to LLM")
    print("    summarize_old_turns=True        # Truncate SQL responses")
    print(")")
    print()


if __name__ == "__main__":
    demo_token_optimization()
