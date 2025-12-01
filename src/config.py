import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_path: str
    quantization: Optional[str]
    gpu_memory_utilization: float
    max_model_len: int
    trust_remote_code: bool
    enforce_eager: bool
    enable_thinking: bool
    enable_openai_fallback: bool
    openai_model: str

@dataclass
class RAGConfig:
    docs_dir: str
    vector_store_dir: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    embeddings_model_openai: str
    embeddings_model_hf: str

@dataclass
class DatabaseConfig:
    path: str
    schema_info: str

@dataclass
class AgentRoleConfig:
    role: str
    system_prompt: str

@dataclass
class AgentConfig:
    guardrails: AgentRoleConfig
    sql: AgentRoleConfig
    analysis: AgentRoleConfig
    viz: AgentRoleConfig
    error: AgentRoleConfig

@dataclass
class AppConfig:
    welcome_message: str

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "configs.yaml")
        self._load_config()
        self._initialized = True

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)
            
        self.llm = LLMConfig(**config_data["llm"])
        self.rag = RAGConfig(**config_data["rag"])
        self.database = DatabaseConfig(**config_data["database"])
        
        agent_data = config_data["agent"]
        self.agent = AgentConfig(
            guardrails=AgentRoleConfig(**agent_data["guardrails"]),
            sql=AgentRoleConfig(**agent_data["sql"]),
            analysis=AgentRoleConfig(**agent_data["analysis"]),
            viz=AgentRoleConfig(**agent_data["viz"]),
            error=AgentRoleConfig(**agent_data["error"])
        )
        
        self.app = AppConfig(**config_data["app"])

# Global accessor
_config_instance = None

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
