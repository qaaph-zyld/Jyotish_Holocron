"""
Utilities for Jyotish Knowledge Base.
Shared functions for config loading, Ollama setup, and common operations.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = str(script_dir / config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_llama_index_settings(config: Dict[str, Any]) -> None:
    """
    Configure LlamaIndex global settings with Ollama models.
    
    Args:
        config: Configuration dictionary
    """
    base_url = config.get("ollama_base_url", "http://127.0.0.1:11434")
    llm_model = config.get("llm_model", "llama3.1:8b")
    embed_model = config.get("embed_model", "nomic-embed-text")
    
    # Set up LLM
    context_window = config.get("llm_context_window", 4096)
    Settings.llm = Ollama(
        model=llm_model,
        base_url=base_url,
        request_timeout=config.get("llm_request_timeout", 180.0),
        context_window=context_window,
    )
    
    # Set up embedding model
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model,
        base_url=base_url,
    )
    
    logging.info(f"Configured LlamaIndex with LLM: {llm_model}, Embed: {embed_model}")


def cap_context(nodes, max_chars: int):
    """
    Cap total retrieved context to prevent timeout/OOM.
    
    Args:
        nodes: List of retrieved nodes
        max_chars: Maximum total characters to include
        
    Returns:
        Capped list of nodes
    """
    total = 0
    kept = []
    for node in nodes:
        txt = node.get_content()
        if total + len(txt) > max_chars:
            break
        kept.append(node)
        total += len(txt)
    return kept


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load ingestion manifest from JSON file.
    
    Args:
        manifest_path: Path to manifest JSON file
        
    Returns:
        Dictionary containing manifest data
    """
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"files": {}, "version": 1}


def save_manifest(manifest: Dict[str, Any], manifest_path: str) -> None:
    """
    Save ingestion manifest to JSON file.
    
    Args:
        manifest: Dictionary containing manifest data
        manifest_path: Path to manifest JSON file
    """
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_markdown_files(data_path: str) -> List[str]:
    """
    Get all markdown files from a path (single file or directory).
    
    Args:
        data_path: Path to markdown file or directory
        
    Returns:
        List of markdown file paths
    """
    path = Path(data_path)
    
    if path.is_file():
        if path.suffix.lower() in ['.md', '.markdown']:
            return [str(path)]
        else:
            raise ValueError(f"File {path} is not a markdown file")
    
    elif path.is_dir():
        md_files = []
        for ext in ['*.md', '*.MD', '*.markdown', '*.MARKDOWN']:
            md_files.extend(path.glob(ext))
        return sorted([str(f) for f in md_files])
    
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")


def check_ollama_connection(base_url: str) -> bool:
    """
    Check if Ollama is accessible.
    
    Args:
        base_url: Ollama base URL
        
    Returns:
        True if Ollama is accessible, False otherwise
    """
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def verify_models_available(base_url: str, llm_model: str, embed_model: str) -> List[str]:
    """
    Verify that required models are available in Ollama.
    
    Args:
        base_url: Ollama base URL
        llm_model: Name of LLM model
        embed_model: Name of embedding model
        
    Returns:
        List of missing models
    """
    import requests
    
    missing = []
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code != 200:
            return [llm_model, embed_model]
        
        data = response.json()
        available_models = [m['name'] for m in data.get('models', [])]
        
        # Check LLM model (handle :tag suffix)
        llm_base = llm_model.split(':')[0]
        if not any(m == llm_model or m.startswith(f"{llm_model}:") or 
                   m.startswith(f"{llm_base}:") for m in available_models):
            missing.append(llm_model)
        
        # Check embedding model
        embed_base = embed_model.split(':')[0]
        if not any(m == embed_model or m.startswith(f"{embed_model}:") or
                   m.startswith(f"{embed_base}:") for m in available_models):
            missing.append(embed_model)
            
    except requests.RequestException as e:
        logging.error(f"Failed to check Ollama models: {e}")
        return [llm_model, embed_model]
    
    return missing


def format_source_info(node, index: int) -> str:
    """
    Format source information from a node for display.
    
    Args:
        node: LlamaIndex Node object
        index: Source index number
        
    Returns:
        Formatted string with source information
    """
    metadata = node.metadata
    source_path = metadata.get('source_path', 'Unknown')
    header_path = metadata.get('header_path', '')
    chunk_idx = metadata.get('chunk_index', 0)
    
    # Get just the filename
    filename = Path(source_path).name if source_path else 'Unknown'
    
    # Build source line
    source_info = f"[{index}] **{filename}**"
    if header_path:
        source_info += f" → `{header_path}`"
    source_info += f" (chunk {chunk_idx})"
    
    return source_info


def truncate_text(text: str, max_chars: int = 300) -> str:
    """
    Truncate text to maximum characters, adding ellipsis if truncated.
    
    Args:
        text: Input text
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated text
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].strip() + "..."
