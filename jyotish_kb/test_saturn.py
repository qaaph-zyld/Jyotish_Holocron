"""Test script to query about Saturn's aspects"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from chat_cli import ChatSession

# Initialize session
session = ChatSession(config_path="config.yaml", verbose=False)

# Test query
query = "What are Saturn's special aspects?"
print(f"\n{'='*60}")
print(f"Query: {query}")
print(f"{'='*60}\n")

# Get answer
response = session.answer(query)
print(response)
