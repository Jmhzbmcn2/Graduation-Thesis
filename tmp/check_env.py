from dotenv import load_dotenv
load_dotenv()
import os

print("=== .env values ===")
print(f"MAX_GLEANING    = {os.getenv('MAX_GLEANING', 'NOT SET')}")
print(f"MAX_ASYNC       = {os.getenv('MAX_ASYNC', 'NOT SET')}")
print(f"MAX_PARALLEL_INSERT = {os.getenv('MAX_PARALLEL_INSERT', 'NOT SET')}")
print(f"LLM_TIMEOUT     = {os.getenv('LLM_TIMEOUT', 'NOT SET')}")
print(f"LLM_MODEL       = {os.getenv('LLM_MODEL', 'NOT SET')}")
print(f"LLM_BINDING     = {os.getenv('LLM_BINDING', 'NOT SET')}")
print(f"ENTITY_TYPES    = {os.getenv('ENTITY_TYPES', 'NOT SET')}")

# Verify types parse correctly
import json
try:
    types = json.loads(os.getenv('ENTITY_TYPES', '[]'))
    print(f"\nParsed ENTITY_TYPES ({len(types)} types):")
    for t in types:
        print(f"  - {t}")
except json.JSONDecodeError as e:
    print(f"\n❌ ENTITY_TYPES JSON ERROR: {e}")
