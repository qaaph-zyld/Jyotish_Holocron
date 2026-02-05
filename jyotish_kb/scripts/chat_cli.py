"""
Interactive CLI chat for Jyotish Knowledge Base.
REPL interface with commands, source tracking, and grounding guardrails.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    cap_context,
    check_ollama_connection,
    format_source_info,
    load_config,
    setup_llama_index_settings,
    setup_logging,
    truncate_text,
    verify_models_available,
)


class ChatSession:
    """
    Interactive chat session with the Jyotish Knowledge Base.
    """
    
    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize chat session.
        
        Args:
            config_path: Path to config YAML
            verbose: Enable verbose logging
        """
        if verbose:
            setup_logging(logging.DEBUG)
        else:
            setup_logging(logging.WARNING)
        
        self.config = load_config(config_path)
        self.config_path = config_path
        
        # Load settings
        self.top_k = self.config.get("top_k", 6)
        self.answer_style = self.config.get("answer_style", "detailed")
        self.reliability_threshold = self.config.get("reliability_threshold", 0.7)
        
        # State
        self.last_sources = []
        self.last_query = ""
        
        # Setup
        self._setup_llm()
        self._load_index()
    
    def _setup_llm(self) -> None:
        """Initialize LLM and embedding models."""
        base_url = self.config.get("ollama_base_url", "http://127.0.0.1:11434")
        
        # Check Ollama
        if not check_ollama_connection(base_url):
            print(f"Error: Cannot connect to Ollama at {base_url}")
            print("Please ensure Ollama is running:")
            print("  ollama serve")
            sys.exit(1)
        
        # Verify models
        llm_model = self.config.get("llm_model", "llama3.1:8b")
        embed_model = self.config.get("embed_model", "nomic-embed-text")
        missing = verify_models_available(base_url, llm_model, embed_model)
        
        if missing:
            print(f"Error: Missing models in Ollama: {', '.join(missing)}")
            print(f"Please pull them:")
            for m in missing:
                print(f"  ollama pull {m}")
            sys.exit(1)
        
        # Setup LlamaIndex
        setup_llama_index_settings(self.config)
    
    def _load_index(self) -> None:
        """Load or create the vector store index."""
        chroma_dir = self.config.get("chroma_dir", "./chroma_db")
        collection_name = self.config.get("collection_name", "jyotish_kb")
        
        # Resolve path
        config_dir = Path(self.config_path).parent if Path(self.config_path).exists() else Path.cwd()
        if not Path(chroma_dir).is_absolute():
            chroma_dir = str(config_dir / chroma_dir)
        
        # Initialize Chroma
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_dir)
            collection = chroma_client.get_collection(name=collection_name)
        except Exception as e:
            print(f"Error: Could not load collection '{collection_name}' from {chroma_dir}")
            print(f"Details: {e}")
            print("\nHave you run the ingestion first?")
            print(f"  python scripts/ingest.py --config {self.config_path}")
            sys.exit(1)
        
        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        
        # Create retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )
        
        logging.info(f"Loaded index with {collection.count()} chunks")
    
    def _update_query_engine(self) -> None:
        """Update query engine with current settings."""
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on answer style."""
        base_prompt = """You are a knowledgeable Vedic astrology (Jyotish) assistant. 
You have access to a comprehensive textbook on Vedic astrology.

Use ONLY the provided context to answer questions. If the context doesn't contain 
sufficient information to answer the question accurately, clearly state that you 
don't have enough information in the knowledge base.

Always cite your sources by referring to the [N] notation provided in the context.
"""
        
        if self.answer_style == "concise":
            base_prompt += """

Provide CONCISE, direct answers. Be brief but accurate.
"""
        else:  # detailed
            base_prompt += """

Provide DETAILED, comprehensive answers. Explain concepts thoroughly with 
examples from the text where relevant. Structure your answer with clear 
headings or sections when appropriate.
"""
        
        return base_prompt
    
    def retrieve(self, query: str) -> Tuple[str, List[Any], float]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (context_text, source_nodes, max_score)
        """
        nodes = self.retriever.retrieve(query)
        
        if not nodes:
            return "", [], 0.0
        
        # Apply prompt budget if configured
        rag_max_chars = self.config.get("rag_max_chars")
        if rag_max_chars and nodes:
            nodes = cap_context(nodes, rag_max_chars)
        
        # Get max score for reliability check
        # Note: ChromaDB returns distances where 0.0 = perfect match
        # Lower distance = better match, so we check the MIN distance
        max_score = min([node.score for node in nodes]) if nodes else 999.0
        
        # Build context with source annotations
        context_parts = []
        for i, node in enumerate(nodes):
            header = node.metadata.get("header_path", "")
            context_parts.append(
                f"[Source {i+1}] {header}\n{node.text}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        return context, nodes, max_score
    
    def answer(self, query: str) -> str:
        """
        Generate an answer to the query using retrieved context.
        
        Args:
            query: User question
            
        Returns:
            Formatted answer with sources
        """
        self.last_query = query
        
        # Retrieve context
        context, nodes, max_score = self.retrieve(query)
        self.last_sources = nodes
        
        # Grounding guardrail
        # ChromaDB uses distance metric: lower = better match
        # Reject if distance is too high (above threshold)
        if not nodes or max_score > self.reliability_threshold:
            return self._weak_retrieval_response(query, nodes, max_score)
        
        # Generate answer using LLM
        system_prompt = self._get_system_prompt()
        
        prompt = f"""{system_prompt}

Question: {query}

Context from Vedic Astrology textbook:
{'='*60}
{context}
{'='*60}

Please answer the question based only on the context provided above.
Include source citations like [1], [2], etc. in your answer.
"""
        
        response = Settings.llm.complete(prompt)
        answer_text = response.text
        
        # Format output
        output = self._format_answer(answer_text, nodes)
        
        return output
    
    def _weak_retrieval_response(self, query: str, nodes: List[Any], max_score: float) -> str:
        """
        Generate response when retrieval is weak.
        
        Args:
            query: Original query
            nodes: Retrieved nodes (may be empty)
            max_score: Highest similarity score
            
        Returns:
            Guardrail response
        """
        response = f"""⚠️ **Limited Information Available**

I couldn't find sufficiently relevant information in the knowledge base for your question:
"{query}"

**Retrieval confidence:** {max_score:.2f} (threshold: {self.reliability_threshold})
"""
        
        if nodes:
            response += "\n**Best matches found (but may not be relevant):**\n"
            for i, node in enumerate(nodes[:3]):
                header = node.metadata.get("header_path", "")
                excerpt = truncate_text(node.text, 150)
                response += f"\n[{i+1}] {header}\n    {excerpt}\n"
        
        response += """\n**Suggestions:**
- Try rephrasing your question with different keywords
- Ask about specific topics covered in the book (e.g., specific grahas, houses, yogas)
- Use more general terms (e.g., "Saturn aspects" instead of "Sade Sati effects")
"""
        
        return response
    
    def _format_answer(self, answer_text: str, nodes: List[Any]) -> str:
        """
        Format the final answer with sources.
        
        Args:
            answer_text: Raw LLM response
            nodes: Source nodes
            
        Returns:
            Formatted answer
        """
        output = "**ANSWER:**\n\n"
        output += answer_text
        output += "\n\n" + "="*60 + "\n"
        output += "**SOURCES:**\n\n"
        
        for i, node in enumerate(nodes):
            source_line = format_source_info(node, i + 1)
            output += f"{source_line}\n"
            
            # Show excerpt
            excerpt = truncate_text(node.text, 200)
            output += f"    \"{excerpt}\"\n\n"
        
        return output
    
    def print_sources(self) -> None:
        """Print full sources from last query."""
        if not self.last_sources:
            print("No sources available. Run a query first.")
            return
        
        print(f"\nSources for: \"{self.last_query}\"\n")
        print("="*60)
        
        for i, node in enumerate(self.last_sources):
            print(f"\n[{i+1}] {format_source_info(node, i+1)}")
            print(f"Full text:\n{node.text}\n")
            print("-"*60)
    
    def print_source(self, index: int) -> None:
        """
        Print a specific source by index.
        
        Args:
            index: 1-based source index
        """
        if not self.last_sources:
            print("No sources available. Run a query first.")
            return
        
        if index < 1 or index > len(self.last_sources):
            print(f"Invalid source number. Range: 1-{len(self.last_sources)}")
            return
        
        node = self.last_sources[index - 1]
        print(f"\n[{index}] {format_source_info(node, index)}")
        print("="*60)
        print(f"\n{node.text}\n")
    
    def handle_command(self, cmd: str) -> bool:
        """
        Handle REPL commands.
        
        Args:
            cmd: Command string (starts with :)
            
        Returns:
            True to continue, False to exit
        """
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if command == ":exit" or command == ":quit":
            print("Goodbye!")
            return False
        
        elif command == ":sources":
            self.print_sources()
        
        elif command == ":open":
            try:
                idx = int(arg.strip())
                self.print_source(idx)
            except ValueError:
                print("Usage: :open <number>")
        
        elif command == ":set":
            self._handle_set(arg)
        
        elif command == ":help":
            self._print_help()
        
        elif command == ":config":
            self._show_config()
        
        else:
            print(f"Unknown command: {command}")
            print("Type :help for available commands")
        
        return True
    
    def _handle_set(self, arg: str) -> None:
        """Handle :set commands."""
        if not arg or '=' not in arg:
            print("Usage: :set <key>=<value>")
            print("  :set top_k=10")
            print("  :set style=detailed")
            return
        
        key, value = arg.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if key == "top_k":
            try:
                self.top_k = int(value)
                self._update_query_engine()
                print(f"top_k set to {self.top_k}")
            except ValueError:
                print("top_k must be an integer")
        
        elif key == "style":
            if value in ["concise", "detailed"]:
                self.answer_style = value
                print(f"style set to {self.answer_style}")
            else:
                print("style must be 'concise' or 'detailed'")
        
        else:
            print(f"Unknown setting: {key}")
    
    def _print_help(self) -> None:
        """Print help message."""
        help_text = """
Jyotish Knowledge Base - Chat Commands
=====================================

Query commands:
  Just type your question and press Enter

Special commands:
  :sources          - Reprint sources from last query
  :open <N>         - Show full text of source N
  :set top_k=N      - Change number of retrieved chunks
  :set style=VALUE  - Change answer style (concise/detailed)
  :config           - Show current configuration
  :help             - Show this help
  :exit / :quit     - Exit the chat

Examples:
  "What are Saturn's aspects?"
  "Explain the 9th house significance"
  "What yogas involve the Sun and Moon?"
        """
        print(help_text)
    
    def _show_config(self) -> None:
        """Show current configuration."""
        print("\nCurrent Configuration:")
        print("="*40)
        print(f"top_k: {self.top_k}")
        print(f"style: {self.answer_style}")
        print(f"LLM: {self.config.get('llm_model', 'default')}")
        print(f"Embed: {self.config.get('embed_model', 'default')}")
        print(f"reliability_threshold: {self.reliability_threshold}")
        print()
    
    def run(self) -> None:
        """Run the interactive REPL."""
        print("\n" + "="*60)
        print("  Jyotish Knowledge Base - Interactive Chat")
        print("="*60)
        print(f"\n  Collection: {self.config.get('collection_name', 'jyotish_kb')}")
        print(f"  LLM: {self.config.get('llm_model', 'default')}")
        print(f"  Answer style: {self.answer_style}")
        print(f"  Top-K retrieval: {self.top_k}")
        print("\n  Type your questions about Vedic Astrology")
        print("  Type :help for commands, :exit to quit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith(":"):
                    if not self.handle_command(user_input):
                        break
                else:
                    # Process query
                    print("\n" + "-"*60)
                    response = self.answer(user_input)
                    print(response)
                    print("-"*60)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type :exit to quit.")
            except EOFError:
                print("\nGoodbye!")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Chat with Jyotish Knowledge Base"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    session = ChatSession(
        config_path=args.config,
        verbose=args.verbose,
    )
    session.run()


if __name__ == "__main__":
    main()
