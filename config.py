import logging
import sqlite3
import openai

# Set logging level to ERROR to minimize unnecessary logs in CMD
logging.basicConfig(level=logging.ERROR)

# Connect to local Ollama (Mistral)
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"  # Dummy key, required by openai-python

# Connect to SQLite DB
conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
