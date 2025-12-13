"""Model-agnostic LLM provider interface."""
import boto3

from abc import ABC, abstractmethod
from typing import Optional
import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text from prompt."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = model
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key or settings.ANTHROPIC_API_KEY)
        self.model = model
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class OllamaProvider(LLMProvider):
    """Local Ollama models."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]


class BedrockProvider(LLMProvider):
    """AWS Bedrock models (Amazon Nova, Claude, etc)."""
    
    def __init__(
        self, 
        model: str = "amazon.nova-lite-v1:0",
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        
        self.model = model
        
        # Initialize boto3 client with credentials from settings
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name or settings.AWS_REGION,
            aws_access_key_id=aws_access_key_id or settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=aws_secret_access_key or settings.AWS_SECRET_ACCESS_KEY,
            aws_session_token=aws_session_token or settings.AWS_SESSION_TOKEN
        )
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import json
        
        # Format request based on model type
        if "nova" in self.model.lower():
            # Amazon Nova models use converse API
            try:
                response = self.client.converse(
                    modelId=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    inferenceConfig={
                        "temperature": temperature,
                        "maxTokens": 2048
                    }
                )
                return response['output']['message']['content'][0]['text']
            except Exception as e:
                # Fallback to invoke_model if converse fails
                request_body = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": temperature,
                        "max_new_tokens": 2048
                    }
                }
                response = self.client.invoke_model(
                    modelId=self.model,
                    body=json.dumps(request_body)
                )
                response_body = json.loads(response['body'].read())
                return response_body['output']['message']['content'][0]['text']
        else:
            # Generic format (Claude, etc)
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']


def create_llm_provider(provider: str = "openai", **kwargs) -> LLMProvider:
    """Factory function to create LLM provider."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "bedrock": BedrockProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")
    
    return providers[provider](**kwargs)
