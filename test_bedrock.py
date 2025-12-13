"""Test script for AWS Bedrock (Amazon Nova Lite) integration."""

import settings
from agent.llm_provider import create_llm_provider


def test_bedrock_connection():
    """Test AWS Bedrock connection with Amazon Nova Lite."""
    
    print("=" * 60)
    print("Testing AWS Bedrock - Amazon Nova Lite")
    print("=" * 60)
    
    # Check if credentials are set
    if not settings.AWS_ACCESS_KEY_ID:
        print("❌ AWS_ACCESS_KEY_ID not found in .env")
        return
    
    if not settings.AWS_SECRET_ACCESS_KEY:
        print("❌ AWS_SECRET_ACCESS_KEY not found in .env")
        return
    
    print("✓ AWS credentials found in .env\n")
    
    try:
        # Create Bedrock provider with Amazon Nova Lite (uses settings.AWS_REGION)
        # Using APAC inference profile for ap-south-1 region
        print(f"Initializing Bedrock provider (region: {settings.AWS_REGION})...")
        print(f"Model: {settings.BEDROCK_MODEL}\n")
        llm = create_llm_provider(
            provider="bedrock",
            model=settings.BEDROCK_MODEL
        )
        print("✓ Bedrock provider initialized\n")
        
        # Test simple query
        print("Sending test query to Amazon Nova Lite...")
        prompt = "What is the capital of France? Answer in one word."
        
        response = llm.generate(prompt, temperature=0.0)
        
        print(f"\n{'=' * 60}")
        print("RESPONSE")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
        print("\n✅ SUCCESS! Amazon Nova Lite is working!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPossible issues:")
        print("1. Check your AWS credentials are correct")
        print("2. Verify your AWS session hasn't expired (temporary credentials)")
        print("3. Ensure you have access to Amazon Nova Lite in Bedrock")
        print("4. Check that you have model access enabled in Bedrock console")


if __name__ == "__main__":
    test_bedrock_connection()
