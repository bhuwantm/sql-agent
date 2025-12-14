"""Centralized prompts and instructions for the SQL agent."""


# System Prompt: Defines the AI's role and persona
# Used in models that support system messages (OpenAI, Anthropic, etc.)
SYSTEM_PROMPT = """You are an expert SQL developer. Generate SQL queries based on business logic and database schemas provided.

Always follow these rules:
- Generate valid SQL queries that fulfill the business logic
- Use proper SQL syntax and best practices
- Include appropriate SQL logical constructs (JOINs, WHERE clauses, GROUP BY, etc.) as needed
- Be explicit about the type of join
- Do not include markdown code blocks or formatting
- Follow the output format instructions exactly"""


# Task Instructions: Specific guidance for the current query
# These are part of the user message/prompt
TASK_INSTRUCTIONS = """## Instructions:
1. Analyze the database schema provided
2. Understand the business logic requirement
3. Generate the appropriate SQL query
4. Ensure the query is optimized and follows best practices"""


# Explanation Instructions: Added when explain=True
EXPLANATION_INSTRUCTIONS = """
After the SQL query, provide a brief explanation of:
1. What tables are being used
2. What the query does
3. Any important joins or conditions"""


def create_sql_prompt(
    business_logic: str,
    schema_context: str,
    explain: bool = False,
    conversation_history: list = None
) -> str:
    """
    Create a complete SQL generation prompt (user message).
    
    Note: This combines everything into a single prompt for simplicity.
    For models supporting system messages, use get_system_prompt() separately.
    
    Args:
        business_logic: Natural language description of what to query
        schema_context: Formatted database schema context
        explain: Whether to include explanation instruction
        conversation_history: List of previous conversation turns (optional)
        
    Returns:
        Complete prompt string
    """
    # Add explanation instructions or explicitly say not to include explanation
    if explain:
        explanation_section = EXPLANATION_INSTRUCTIONS
        output_format = "Return the SQL query followed by the explanation."
    else:
        explanation_section = ""
        output_format = "Return ONLY the SQL query. Do not include any explanation, description, or additional text."
    
    # Format conversation history if provided (already optimized by caller)
    history_context = ""
    if conversation_history:
        history_context = "\n## Previous Conversation:\n"
        for i, turn in enumerate(conversation_history, 1):
            history_context += f"\nTurn {i}:\n"
            history_context += f"User: {turn['user']}\n"
            history_context += f"Assistant: {turn['assistant']}\n"
        history_context += "\n## Current Request:\n"
    
    # Combine system prompt + user message for single-prompt models
    prompt = f"""{SYSTEM_PROMPT}

## Database Schema (Relevant Tables Only):
{schema_context}
{history_context}
## Business Logic:
{business_logic}

{TASK_INSTRUCTIONS}
{explanation_section}

## Output Format:
{output_format}

SQL Query:"""
    
    return prompt


def get_system_prompt() -> str:
    """
    Get the system prompt separately.
    
    Use this for models that support dedicated system messages
    (e.g., OpenAI's system role, Anthropic's system parameter).
    
    Returns:
        System prompt string
    """
    return SYSTEM_PROMPT


def create_user_message(
    business_logic: str,
    schema_context: str,
    explain: bool = False
) -> str:
    """
    Create just the user message part (without system prompt).
    
    Use this when you want to send system and user messages separately.
    
    Args:
        business_logic: Natural language description of what to query
        schema_context: Formatted database schema context
        explain: Whether to include explanation instruction
        
    Returns:
        User message string
    """
    explanation_section = EXPLANATION_INSTRUCTIONS if explain else ""
    
    user_message = f"""## Database Schema (Relevant Tables Only):
{schema_context}

## Business Logic:
{business_logic}

{TASK_INSTRUCTIONS}
{explanation_section}

SQL Query:"""
    
    return user_message
