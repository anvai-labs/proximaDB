#!/usr/bin/env python3
"""
LLM Service for RAG Demo
Provides lightweight LLM capabilities with multiple fallback options
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    model_type: str = "flan-t5-small"  # or "api", "none"
    api_provider: str = "huggingface"  # or "groq", "together"
    api_key: Optional[str] = None
    max_length: int = 150
    temperature: float = 0.7
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set default cache directory based on environment"""
        if self.cache_dir is None:
            # Check if running in Docker
            if Path("/app/embedding_cache").exists():
                self.cache_dir = Path("/app/embedding_cache/llm")
            else:
                # Use local cache directory
                self.cache_dir = Path.home() / ".cache" / "proximadb" / "llm"


class LLMService:
    """Lightweight LLM service for RAG enhancement"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to initialize based on config
        self._initialize()
    
    def _initialize(self):
        """Initialize the LLM based on configuration"""
        try:
            if self.config.model_type == "flan-t5-small":
                self._init_local_model()
            elif self.config.model_type == "api":
                self._init_api_client()
            else:
                logger.info("LLM disabled, using template-based responses")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}. Falling back to templates.")
            self.config.model_type = "none"
    
    def _init_local_model(self):
        """Initialize local Flan-T5 model"""
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model_name = "google/flan-t5-small"
            logger.info(f"Loading {model_name} model...")
            
            # Download and cache the model
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Move to CPU and eval mode
            self.model.eval()
            logger.info("✅ Flan-T5 model loaded successfully")
            
        except ImportError:
            logger.warning("Transformers not available. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _init_api_client(self):
        """Initialize API client for external LLM services"""
        if self.config.api_provider == "huggingface":
            # Hugging Face Inference API - try without authentication first
            self.api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
            logger.info("Using Hugging Face Inference API (attempting free access)")
        elif self.config.api_provider == "groq":
            # Groq requires API key
            if not self.config.api_key:
                raise ValueError("Groq API requires an API key")
            self.api_url = "https://api.groq.com/v1/completions"
            logger.info("Using Groq API")
        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}")
    
    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_length: Optional[int] = None
    ) -> str:
        """Generate a response using the LLM"""
        if not self._initialized:
            return self._template_response(query, context)
        
        # Format the prompt for RAG
        prompt = self._format_rag_prompt(query, context)
        
        try:
            if self.config.model_type == "flan-t5-small" and self.model:
                return self._generate_local(prompt, max_length)
            elif self.config.model_type == "api":
                return self._generate_api(prompt, max_length)
            else:
                return self._template_response(query, context)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._template_response(query, context)
    
    def _format_rag_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format the RAG prompt for the LLM"""
        # Build context string from search results
        context_str = "\n\n".join([
            f"Document {i+1}:\n{doc.get('content', doc.get('text', ''))[:500]}"
            for i, doc in enumerate(context[:3])  # Use top 3 results
        ])
        
        # Flan-T5 works well with instruction-style prompts
        prompt = f"""Answer the question based on the following context:

Context:
{context_str}

Question: {query}

Answer:"""
        
        return prompt
    
    def _generate_local(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate response using local model"""
        max_length = max_length or self.config.max_length
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate response
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=self.config.temperature,
            do_sample=True,
            top_p=0.95,
            num_beams=2
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _generate_api(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate response using API"""
        max_length = max_length or self.config.max_length
        
        if self.config.api_provider == "huggingface":
            # Hugging Face Inference API
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": self.config.temperature,
                    "do_sample": True
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").replace(prompt, "").strip()
                return str(result)
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise Exception(f"API request failed: {response.status_code}")
        
        # Add other API providers here
        raise NotImplementedError(f"API provider {self.config.api_provider} not implemented")
    
    def _template_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Fallback template-based response when LLM is not available"""
        if not context:
            return "I couldn't find any relevant information to answer your question."
        
        # Simple template-based response
        top_result = context[0]
        content = top_result.get('content', top_result.get('text', ''))[:200]
        
        response = f"Based on the search results, here's what I found:\n\n{content}..."
        
        if len(context) > 1:
            response += f"\n\nI found {len(context)} relevant documents that might help answer your question."
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_type": self.config.model_type,
            "initialized": self._initialized,
            "provider": self.config.api_provider if self.config.model_type == "api" else "local",
            "model_name": "google/flan-t5-small" if self.config.model_type == "flan-t5-small" else "none"
        }


# Singleton instance
_llm_service = None

def get_llm_service(config: Optional[LLMConfig] = None) -> LLMService:
    """Get or create the LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(config)
    return _llm_service


# Example usage
if __name__ == "__main__":
    # Test the service
    service = get_llm_service()
    
    # Example context from vector search
    context = [
        {"content": "ProximaDB is a high-performance vector database designed for AI applications."},
        {"content": "It supports multiple storage engines including NOVA and SST."},
        {"content": "ProximaDB offers both REST and gRPC APIs for flexible integration."}
    ]
    
    # Generate response
    response = service.generate_response(
        query="What is ProximaDB?",
        context=context
    )
    
    print(f"Model info: {service.get_model_info()}")
    print(f"Response: {response}")