"""
vLLM Manager for Local Inference
Handles loading the Qwen model and providing a unified interface for generation,
with optional fallback to OpenAI.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import time

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ vLLM not installed. Running in API-only mode or mock mode.")

from openai import OpenAI
from monitoring import monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool
    quantization: Optional[str] = None
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 8192

class VLLMManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLLMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.llm = None
        self.tokenizer = None
        self.openai_client = None
        
        # Configuration
        self.local_model_config = ModelConfig(
            name="Qwen/Qwen3-8B-AWQ",
            model_path="Qwen/Qwen3-8B-AWQ",
            is_local=True,
            quantization="awq",
            gpu_memory_utilization=0.8,
            max_model_len=32768 
        )
        
        self.enable_openai_fallback = os.getenv("ENABLE_OPENAI_FALLBACK", "true").lower() == "true"
        
        self._initialize_models()
        self._initialized = True

    def _initialize_models(self):
        """Initialize vLLM and/or OpenAI client"""
        
        # Initialize OpenAI client if fallback enabled
        if self.enable_openai_fallback and os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("✅ OpenAI client initialized for fallback")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")

        # Initialize vLLM
        if VLLM_AVAILABLE:
            try:
                logger.info(f"🚀 Initializing vLLM with model: {self.local_model_config.model_path}")
                
                self.llm = LLM(
                    model=self.local_model_config.model_path,
                    quantization=self.local_model_config.quantization,
                    gpu_memory_utilization=self.local_model_config.gpu_memory_utilization,
                    max_model_len=self.local_model_config.max_model_len,
                    trust_remote_code=True,
                    enforce_eager=False
                )
                
                # Get tokenizer for chat template formatting
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.local_model_config.model_path, 
                    trust_remote_code=True
                )
                
                logger.info("✅ vLLM initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize vLLM: {e}")
                self.llm = None
        else:
            logger.warning("⚠️ vLLM not available. Will rely on OpenAI fallback if enabled.")

    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate response using local model with fallback to OpenAI.
        Accepts messages in standard chat format: [{"role": "user", "content": "..."}]
        """
        
        # Try local generation first
        if self.llm and self.tokenizer:
            try:
                return self._generate_local(messages, temperature, max_tokens, stop)
            except Exception as e:
                logger.error(f"❌ Local generation failed: {e}")
                if not self.enable_openai_fallback:
                    raise e
                logger.info("🔄 Falling back to OpenAI...")

        # Fallback to OpenAI
        if self.enable_openai_fallback and self.openai_client:
            return self._generate_openai(messages, temperature, max_tokens, response_format)
        
        raise RuntimeError("No available LLM backend (Local failed/missing and OpenAI disabled/missing)")

    def _generate_local(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate using vLLM"""
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
            top_p=0.95
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text

    def _generate_openai(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict] = None
    ) -> str:
        """Generate using OpenAI"""
        
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            kwargs["response_format"] = response_format
            
        response = self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

# Global accessor
def get_vllm_manager():
    return VLLMManager()
