import os
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()

# ChromaDB Settings
CHROMA_DIRECTORY = "chroma_db"
CHROMA_COLLECTION_NAME = "database_schemas"

# Directory where database schema table JSON files are stored
DATABASE_SCHEMAS_DIR = "schemas"

# AWS Bedrock Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "apac.amazon.nova-lite-v1:0")

# LLM API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
