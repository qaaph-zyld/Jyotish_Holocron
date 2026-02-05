"""Test script with exact text from book"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from chat_cli import ChatSession

# Initialize session
session = ChatSession(config_path="config.yaml", verbose=False)

# Test queries
queries = [
    "Saturn aspects the 3rd and 10th houses from him",
    "special aspects of Mars Jupiter and Saturn",
    "planetary aspects in Vedic astrology",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    response = session.answer(query)
    print(response)
    print("\n" + "="*60 + "\n")
