import json
import os

# Xem thong tin file goc
data_file = r'C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\data\acepron.txt'
if os.path.exists(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    print('='*60)
    print('DATA FILE ANALYSIS')
    print('='*60)
    print(f'File path: {data_file}')
    print(f'File size: {len(content)} characters')
    print(f'Word count: ~{len(content.split())} words')
    
    # Uoc tinh tokens (rough: 1 token ~ 4 chars for Vietnamese/English mix)
    est_tokens = len(content) // 4
    print(f'Estimated tokens: ~{est_tokens} tokens')
    print()
    print('Cau hinh chunking (trong script):')
    print('  - chunk_token_size: 1200 tokens')
    print('  - chunk_overlap_token_size: 100 tokens')
    print()
    expected_chunks = max(1, est_tokens // (1200 - 100))
    print(f'So chunks du kien: ~{expected_chunks}')
else:
    print(f'File khong ton tai: {data_file}')

# Xem chunks da luu
chunks_file = './acepron_rag/kv_store_text_chunks.json'
if os.path.exists(chunks_file):
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print()
    print('='*60)
    print('CHUNKS STORED')
    print('='*60)
    print(f'Total chunks: {len(chunks_data)}')
    print()
    
    for chunk_id, chunk_info in chunks_data.items():
        tokens = chunk_info.get('tokens', 'N/A')
        print(f'Chunk: {chunk_id}')
        print(f'  Tokens: {tokens}')
        print()

# Xem full docs
docs_file = './acepron_rag/kv_store_full_docs.json'
if os.path.exists(docs_file):
    with open(docs_file, 'r', encoding='utf-8') as f:
        docs_data = json.load(f)
    
    print('='*60)
    print('FULL DOCUMENTS')
    print('='*60)
    print(f'Total documents: {len(docs_data)}')
    for doc_id, doc_info in docs_data.items():
        content = doc_info.get('content', '')
        print(f'Doc: {doc_id}')
        print(f'  Length: {len(content)} chars')
