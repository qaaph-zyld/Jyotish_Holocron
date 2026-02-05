"""Test retrieval without LLM to show the system is working"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from chat_cli import ChatSession

# Initialize session
session = ChatSession(config_path="config.yaml", verbose=False)

# Test queries
queries = [
    "What are Saturn's special aspects?",
    "What are the references about Mrigasira nakshatra?",
    "planetary aspects in Vedic astrology",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    
    # Get retrieval results only (no LLM)
    context, nodes, best_distance = session.retrieve(query)
    
    print(f"\n📊 Retrieval Results:")
    print(f"  - Best match distance: {best_distance:.4f}")
    print(f"  - Threshold: {session.reliability_threshold}")
    print(f"  - Number of chunks retrieved: {len(nodes)}")
    print(f"  - Status: {'✅ PASS' if best_distance <= session.reliability_threshold else '❌ FAIL (no relevant content)'}")
    
    if nodes and best_distance <= session.reliability_threshold:
        print(f"\n📚 Top 3 Sources:")
        for i, node in enumerate(nodes[:3], 1):
            header = node.metadata.get("header_path", "/")
            distance = node.score
            excerpt = node.text[:200].replace('\n', ' ')
            
            print(f"\n  [{i}] Distance: {distance:.4f}")
            print(f"      Header: {header}")
            print(f"      Text: {excerpt}...")
    
    print()
