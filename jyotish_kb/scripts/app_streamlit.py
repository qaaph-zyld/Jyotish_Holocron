"""
Streamlit web UI for Jyotish Knowledge Base.
Provides a chat interface with sidebar settings and source citations.
"""

import logging
import sys
from pathlib import Path

import chromadb
import streamlit as st
from llama_index.core import Settings, StorageContext, VectorStoreIndex
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
    truncate_text,
    verify_models_available,
)

# Page config
st.set_page_config(
    page_title="Jyotish Knowledge Base",
    page_icon="🕉️",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []


def init_knowledge_base(config_path: str):
    """Initialize or retrieve the knowledge base from session state."""
    if "kb_initialized" not in st.session_state:
        # Load config
        config = load_config(config_path)
        
        # Check Ollama
        base_url = config.get("ollama_base_url", "http://127.0.0.1:11434")
        if not check_ollama_connection(base_url):
            st.error(f"❌ Cannot connect to Ollama at {base_url}")
            st.info("Please ensure Ollama is running:")
            st.code("ollama serve", language="bash")
            return None
        
        # Verify models
        llm_model = config.get("llm_model", "llama3.1:8b")
        embed_model = config.get("embed_model", "nomic-embed-text")
        missing = verify_models_available(base_url, llm_model, embed_model)
        
        if missing:
            st.error(f"❌ Missing models: {', '.join(missing)}")
            st.info("Please pull the required models:")
            for m in missing:
                st.code(f"ollama pull {m}", language="bash")
            return None
        
        # Setup LlamaIndex
        setup_llama_index_settings(config)
        
        # Load ChromaDB
        chroma_dir = config.get("chroma_dir", "./chroma_db")
        collection_name = config.get("collection_name", "jyotish_kb")
        
        # Resolve path
        config_dir = Path(config_path).parent if Path(config_path).exists() else Path.cwd()
        if not Path(chroma_dir).is_absolute():
            chroma_dir = str(config_dir / chroma_dir)
        
        try:
            chroma_client = chromadb.PersistentClient(path=chroma_dir)
            collection = chroma_client.get_collection(name=collection_name)
        except Exception as e:
            st.error(f"❌ Could not load knowledge base")
            st.info(f"Details: {e}")
            st.info("Have you run the ingestion first?")
            st.code(f"python scripts/ingest.py --config {config_path}", language="bash")
            return None
        
        # Create index and retriever
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        
        st.session_state.config = config
        st.session_state.index = index
        st.session_state.collection = collection
        st.session_state.kb_initialized = True
        
        return config
    
    return st.session_state.config


def retrieve_context(index, query: str, top_k: int):
    """Retrieve relevant chunks for a query."""
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    
    nodes = retriever.retrieve(query)
    
    if not nodes:
        return [], "", 0.0
    
    # Apply prompt budget if configured
    config = st.session_state.config
    rag_max_chars = config.get("rag_max_chars")
    if rag_max_chars and nodes:
        nodes = cap_context(nodes, rag_max_chars)
    
    max_score = max([node.score for node in nodes]) if nodes else 0.0
    
    # Build context
    context_parts = []
    for i, node in enumerate(nodes):
        header = node.metadata.get("header_path", "")
        context_parts.append(
            f"[Source {i+1}] {header}\n{node.text}\n"
        )
    
    context = "\n---\n".join(context_parts)
    
    return nodes, context, max_score


def generate_answer(config, query: str, context: str, sources: list) -> str:
    """Generate an answer using the LLM."""
    style = config.get("answer_style", "detailed")
    
    system_prompt = """You are a knowledgeable Vedic astrology (Jyotish) assistant. 
You have access to a comprehensive textbook on Vedic astrology.

Use ONLY the provided context to answer questions. If the context doesn't contain 
sufficient information to answer the question accurately, clearly state that you 
don't have enough information in the knowledge base.

Always cite your sources by referring to the [N] notation provided in the context.
"""
    
    if style == "concise":
        system_prompt += "\nProvide CONCISE, direct answers. Be brief but accurate.\n"
    else:
        system_prompt += "\nProvide DETAILED, comprehensive answers with examples.\n"
    
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
    return response.text


def main():
    # Title
    st.title("🕉️ Jyotish Knowledge Base")
    st.markdown("Chat with your Vedic Astrology textbook using local AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        config_path = st.text_input(
            "Config file",
            value="config.yaml",
            help="Path to configuration YAML file"
        )
        
        top_k = st.slider(
            "Top-K retrieval",
            min_value=1,
            max_value=20,
            value=6,
            help="Number of chunks to retrieve"
        )
        
        style = st.selectbox(
            "Answer style",
            options=["detailed", "concise"],
            index=0,
            help="Detailed provides comprehensive answers, concise is brief"
        )
        
        reliability_threshold = st.slider(
            "Reliability threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for confident answers"
        )
        
        st.divider()
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()
    
    # Initialize KB
    config = init_knowledge_base(config_path)
    
    if config is None:
        st.stop()
    
    # Update config with sidebar values
    config["top_k"] = top_k
    config["answer_style"] = style
    config["reliability_threshold"] = reliability_threshold
    
    # Show KB info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collection", config.get("collection_name", "jyotish_kb"))
    with col2:
        st.metric("Chunks", st.session_state.collection.count())
    with col3:
        st.metric("LLM", config.get("llm_model", "default").split(":")[0])
    
    # Chat interface
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about Vedic Astrology..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                nodes, context, max_score = retrieve_context(
                    st.session_state.index,
                    prompt,
                    top_k
                )
                st.session_state.sources = nodes
            
            # Check reliability
            if not nodes or max_score < reliability_threshold:
                response = f"""⚠️ **Limited Information Available**

I couldn't find sufficiently relevant information in the knowledge base for your question.

**Retrieval confidence:** {max_score:.2f} (threshold: {reliability_threshold})

**Suggestions:**
- Try rephrasing your question
- Ask about specific topics covered in the book
- Use terms from Vedic astrology (grahas, houses, yogas, etc.)
"""
                if nodes:
                    response += "\n**Best matches found:**\n"
                    for i, node in enumerate(nodes[:3]):
                        header = node.metadata.get("header_path", "")
                        excerpt = truncate_text(node.text, 100)
                        response += f"\n[{i+1}] {header}: {excerpt}\n"
                
                st.markdown(response)
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(config, prompt, context, nodes)
                
                st.markdown(answer)
                
                # Sources expander
                with st.expander("📚 View Sources"):
                    for i, node in enumerate(nodes):
                        source_info = format_source_info(node, i + 1)
                        st.markdown(f"**{source_info}**")
                        st.text_area(
                            f"Source {i+1} text",
                            value=node.text,
                            height=150,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        st.divider()
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer if nodes and max_score >= reliability_threshold else response
        })


if __name__ == "__main__":
    main()
