"""
Ingestion script for Jyotish Knowledge Base.
Parses markdown files, creates chunks with metadata, and stores in ChromaDB.
Supports incremental updates based on file hashes.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    check_ollama_connection,
    compute_file_hash,
    get_markdown_files,
    load_config,
    load_manifest,
    save_manifest,
    setup_llama_index_settings,
    setup_logging,
    verify_models_available,
)


def create_chunks_with_metadata(
    file_path: str,
    chunk_size: int = 900,
    chunk_overlap: int = 100,
) -> List[TextNode]:
    """
    Create chunks from a markdown file with header path metadata.
    
    Uses MarkdownNodeParser to split by headers first, then SentenceSplitter
    for any nodes that are still too large.
    
    Args:
        file_path: Path to markdown file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of TextNode objects with metadata
    """
    logging.info(f"Processing {file_path}...")
    
    # Load document
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    document = Document(text=content, metadata={"source_path": file_path})
    
    # Step 1: Parse with MarkdownNodeParser to preserve header structure
    md_parser = MarkdownNodeParser()
    md_nodes = md_parser.get_nodes_from_documents([document])
    
    logging.info(f"  MarkdownNodeParser created {len(md_nodes)} nodes")
    
    # Step 2: Apply SentenceSplitter for any nodes that are too large
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    final_nodes = []
    for i, node in enumerate(md_nodes):
        # Get header path from existing metadata
        header_path = node.metadata.get("header_path", "")
        
        # Check if this node needs further splitting
        # Rough estimate: 1 token ~= 4 characters for English
        estimated_tokens = len(node.text) / 4
        
        if estimated_tokens > chunk_size * 1.2:  # Allow 20% buffer
            # Split this node further
            sub_nodes = sentence_splitter.get_nodes_from_documents(
                [Document(text=node.text, metadata=node.metadata)]
            )
            logging.debug(f"    Split node {i} ({len(node.text)} chars) into {len(sub_nodes)} chunks")
            
            for j, sub_node in enumerate(sub_nodes):
                sub_node.metadata["source_path"] = file_path
                sub_node.metadata["header_path"] = header_path
                sub_node.metadata["chunk_index"] = len(final_nodes)
                sub_node.metadata["parent_node"] = i
                final_nodes.append(sub_node)
        else:
            # Keep as-is
            node.metadata["source_path"] = file_path
            node.metadata["header_path"] = header_path
            node.metadata["chunk_index"] = len(final_nodes)
            final_nodes.append(node)
    
    logging.info(f"  Final chunk count: {len(final_nodes)}")
    return final_nodes


def ingest_files_to_index(
    file_paths: List[str],
    vector_store: ChromaVectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Ingest multiple files into the LlamaIndex vector store.
    
    Args:
        file_paths: List of markdown file paths
        vector_store: ChromaDB vector store
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        
    Returns:
        Dictionary mapping file paths to ingestion info
    """
    all_nodes = []
    file_info = {}
    
    for file_path in file_paths:
        file_hash = compute_file_hash(file_path)
        
        # Create chunks
        nodes = create_chunks_with_metadata(file_path, chunk_size, chunk_overlap)
        
        # Add file hash to each node's metadata
        for node in nodes:
            node.metadata["file_hash"] = file_hash
        
        all_nodes.extend(nodes)
        
        # Track info for manifest
        rel_path = str(Path(file_path).name)
        file_info[rel_path] = {
            "hash": file_hash,
            "timestamp": datetime.now().isoformat(),
            "num_chunks": len(nodes),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "source_path": file_path,
        }
        
        logging.info(f"  Prepared {len(nodes)} chunks from {rel_path}")
    
    # Create index with all nodes - this generates embeddings
    if all_nodes:
        logging.info(f"\nGenerating embeddings for {len(all_nodes)} chunks...")
        logging.info(f"Using embedding model: {Settings.embed_model.model_name}")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index from nodes - this creates embeddings
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        
        logging.info(f"Successfully indexed {len(all_nodes)} chunks")
    
    return file_info


def delete_file_entries(collection: Any, file_path: str) -> int:
    """
    Delete all entries for a specific file from ChromaDB.
    
    Args:
        collection: ChromaDB collection
        file_path: Path to the file to delete
        
    Returns:
        Number of deleted entries
    """
    try:
        results = collection.get(
            where={"source_path": file_path}
        )
        
        if results and 'ids' in results and results['ids']:
            ids_to_delete = results['ids']
            collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
    except Exception as e:
        logging.warning(f"Could not delete entries for {file_path}: {e}")
    
    return 0


def run_ingestion(
    config_path: str,
    rebuild: bool = False,
    verbose: bool = False,
) -> None:
    """
    Main ingestion pipeline.
    
    Args:
        config_path: Path to config YAML
        rebuild: If True, wipe DB and rebuild from scratch
        verbose: Enable verbose logging
    """
    if verbose:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)
    
    # Load config
    config = load_config(config_path)
    
    # Paths
    data_path = config.get("data_path", "./data")
    chroma_dir = config.get("chroma_dir", "./chroma_db")
    collection_name = config.get("collection_name", "jyotish_kb")
    chunk_size = config.get("chunk_size_tokens", 900)
    chunk_overlap = config.get("chunk_overlap_tokens", 100)
    
    # Resolve paths relative to config location
    config_dir = Path(config_path).parent if Path(config_path).exists() else Path.cwd()
    if not Path(data_path).is_absolute():
        data_path = str(config_dir / data_path)
    if not Path(chroma_dir).is_absolute():
        chroma_dir = str(config_dir / chroma_dir)
    
    manifest_path = Path(chroma_dir) / "manifest.json"
    
    logging.info(f"Data path: {data_path}")
    logging.info(f"Chroma directory: {chroma_dir}")
    logging.info(f"Collection: {collection_name}")
    
    # Check Ollama
    base_url = config.get("ollama_base_url", "http://127.0.0.1:11434")
    if not check_ollama_connection(base_url):
        logging.error(f"Cannot connect to Ollama at {base_url}")
        logging.error("Please ensure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Verify models
    llm_model = config.get("llm_model", "llama3.1:8b")
    embed_model = config.get("embed_model", "nomic-embed-text")
    missing = verify_models_available(base_url, llm_model, embed_model)
    if missing:
        logging.error(f"Missing models in Ollama: {', '.join(missing)}")
        logging.error(f"Please pull them: ollama pull {' && ollama pull '.join(missing)}")
        sys.exit(1)
    
    # Setup LlamaIndex BEFORE any vector operations
    setup_llama_index_settings(config)
    
    # Initialize ChromaDB
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    
    if rebuild:
        logging.info("REBUILD mode: wiping existing database...")
        try:
            chroma_client.delete_collection(name=collection_name)
            logging.info(f"Deleted collection: {collection_name}")
        except Exception:
            pass
        manifest = {"files": {}, "version": 1}
    else:
        manifest = load_manifest(str(manifest_path))
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
        logging.info(f"Using existing collection: {collection_name}")
        
        if rebuild:
            chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(name=collection_name)
            logging.info(f"Created fresh collection: {collection_name}")
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
        logging.info(f"Created new collection: {collection_name}")
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    # Get markdown files
    try:
        md_files = get_markdown_files(data_path)
    except FileNotFoundError as e:
        logging.error(f"Data path not found: {e}")
        sys.exit(1)
    
    if not md_files:
        logging.warning(f"No markdown files found in {data_path}")
        sys.exit(0)
    
    logging.info(f"Found {len(md_files)} markdown file(s)")
    
    # Determine which files need processing
    files_to_process = []
    files_skipped = 0
    files_updated = 0
    
    for file_path in md_files:
        file_hash = compute_file_hash(file_path)
        rel_path = str(Path(file_path).name)
        
        # Check if already ingested and unchanged
        if rel_path in manifest["files"]:
            if manifest["files"][rel_path]["hash"] == file_hash and not rebuild:
                logging.info(f"Skipping (unchanged): {rel_path}")
                files_skipped += 1
                continue
            else:
                logging.info(f"Updating (changed): {rel_path}")
                # Delete old entries
                deleted = delete_file_entries(collection, file_path)
                logging.info(f"  Deleted {deleted} old chunks")
                files_updated += 1
        else:
            logging.info(f"Ingesting: {rel_path}")
            files_updated += 1
        
        files_to_process.append(file_path)
    
    # Process files if any
    if files_to_process:
        logging.info(f"\nProcessing {len(files_to_process)} file(s)...")
        
        new_file_info = ingest_files_to_index(
            files_to_process,
            vector_store,
            chunk_size,
            chunk_overlap,
        )
        
        # Update manifest
        manifest["files"].update(new_file_info)
    else:
        logging.info("No files to process")
    
    # Save manifest
    save_manifest(manifest, str(manifest_path))
    
    # Summary
    logging.info("=" * 50)
    logging.info(f"Ingestion complete!")
    logging.info(f"  New/updated files: {files_updated}")
    logging.info(f"  Skipped (unchanged): {files_skipped}")
    logging.info(f"  Total chunks in DB: {collection.count()}")
    logging.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest markdown files into Jyotish Knowledge Base"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe database and rebuild from scratch",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    run_ingestion(
        config_path=args.config,
        rebuild=args.rebuild,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
