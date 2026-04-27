import os
import json
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from rank_bm25 import BM25Okapi
import time
import random
from dotenv import load_dotenv

load_dotenv()

# Tệp dữ liệu câu hỏi
CSV_PATH = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\testcase\300_case_random.csv"

# Tệp Vector DB có sẵn của LightRAG
VDB_PATH = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_rag\medical_rag_v2\vdb_entities.json"

# Thiết lập LLM để extract keyword (Dùng OpenRouter)
client = OpenAI(
    api_key=os.environ.get("LLM_BINDING_API_KEY", "YOUR_OPENROUTER_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Thiết lập Client cho Ollama local để nhúng query
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)

def get_ollama_embedding(text: str) -> list:
    """Gọi API của Ollama local thông qua OpenAI client tương thích để lấy vector embedding."""
    try:
        response = ollama_client.embeddings.create(
            model="embeddinggemma:300m",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Lỗi khi gọi Ollama Embedding: {e}")
        return [0.0] * 768 

def extract_keywords(query: str) -> list:
    """Gọi LLM để extract low_level_keywords từ câu hỏi."""
    system_prompt = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object. Do not include markdown code fences (like ```json).
2. **Language**: All extracted keywords MUST be in Vietnamese.
"""
    user_prompt = f"---Real Data---\nUser Query: {query}\n---Output---\nOutput:"

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-30b-a3b-instruct-2507",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        result = json.loads(content)
        return result.get("low_level_keywords", [])
    except Exception as e:
        print("Lỗi khi parse JSON từ LLM:", e)
        return []

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def get_ground_truth_rank(ground_truth_name, retrieved_names):
    # Kiểm tra xem tên ground truth (từ CSV) có nằm trong tên entity tìm được hay không
    # Trả về thứ hạng (1-based index). Nếu không tìm thấy trả về 0.
    gt = str(ground_truth_name).lower().strip()
    for i, name in enumerate(retrieved_names):
        if gt in name.lower() or name.lower() in gt:
            return i + 1
    return 0

import base64
import zlib

def parse_vector(v):
    if v is None:
        return [0.0] * 768
    if isinstance(v, str):
        try:
            # LightRAG lưu vector dưới dạng nén zlib + base64 + np.float16/float32
            data = zlib.decompress(base64.b64decode(v))
            if len(data) == 768 * 2:
                arr = np.frombuffer(data, dtype=np.float16)
            elif len(data) == 768 * 4:
                arr = np.frombuffer(data, dtype=np.float32)
            else:
                arr = np.frombuffer(data, dtype=np.float32) # fallback
            return arr.astype(float).tolist()
        except:
            try:
                import ast
                v_list = ast.literal_eval(v) if '[' in v else json.loads(v)
                return [float(x) for x in v_list]
            except:
                return [0.0] * 768
    if isinstance(v, list) or isinstance(v, tuple):
        return [float(x) for x in v]
    return [0.0] * 768

def evaluate_search_methods(sample_size=10):
    if not os.path.exists(VDB_PATH):
        print(f"Không tìm thấy file VDB tại {VDB_PATH}.")
        return

    # 1. Tải toàn bộ Entity từ Knowledge Graph có sẵn
    print("Đang đọc và tải dữ liệu từ Knowledge Graph (vdb_entities.json)...")
    start_time = time.time()
    with open(VDB_PATH, 'r', encoding='utf-8') as f:
        vdb_data = json.load(f)
    
    entities = vdb_data.get('data', [])
    names = [str(ent.get('entity_name', '')) for ent in entities]
    descriptions = [str(ent.get('content', '')) for ent in entities]
    
    # Ép kiểu an toàn cho vector (đề phòng JSON lưu vector dưới dạng chuỗi hoặc list hỗn hợp)
    doc_embeddings = [np.array(parse_vector(ent.get('vector')), dtype=float) for ent in entities]
    print(f"✅ Đã tải xong {len(entities)} entities từ database trong {time.time() - start_time:.1f}s!")

    # 2. Khởi tạo BM25 cho Name (Từ Graph)
    print("Đang khởi tạo BM25 từ tên các Entity trong Graph...")
    tokenized_names = [name.lower().split() for name in names]
    bm25 = BM25Okapi(tokenized_names)
    
    # 3. Đọc dữ liệu câu hỏi từ CSV
    if not os.path.exists(CSV_PATH):
        print(f"Không tìm thấy file {CSV_PATH}.")
        return

    df = pd.read_csv(CSV_PATH)
    questions = df['question'].fillna("").tolist()
    csv_keywords = df['keyword'].fillna("").tolist() # Ground truth
    
    test_indices = random.sample(range(len(questions)), min(sample_size, len(questions)))
    
    vector_hits = 0
    bm25_hits = 0
    vector_mrr_sum = 0.0
    bm25_mrr_sum = 0.0
    vector_time_sum = 0.0
    bm25_time_sum = 0.0
    top_k = 5
    
    print("\n" + "="*60)
    print(f"BẮT ĐẦU ĐÁNH GIÁ CHẤT LƯỢNG TRÊN {len(test_indices)} CÂU HỎI MẪU (Sử dụng DB Thực Tế)")
    print("="*60)
    
    for idx in test_indices:
        query = questions[idx]
        ground_truth_name = csv_keywords[idx]
        
        print(f"\n[Câu hỏi]: {query}")
        print(f"[Ground Truth (Từ CSV)]: {ground_truth_name}")
        
        ll_keywords = extract_keywords(query)
        print(f" => LL_Keywords trích xuất được: {ll_keywords}")
        
        if not ll_keywords:
            print(" => Bỏ qua vì không extract được keyword.")
            continue
            
        search_query_str = " ".join(ll_keywords)
        
        # Phương pháp 1: Vector Search trên DB thực
        t_vector_start = time.time()
        raw_query_emb = get_ollama_embedding(search_query_str)
        query_emb = np.array(parse_vector(raw_query_emb), dtype=float)
        
        similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
        top_vector_indices = np.argsort(similarities)[::-1][:top_k]
        vector_retrieved_names = [names[i] for i in top_vector_indices]
        t_vector_end = time.time()
        
        vector_time = t_vector_end - t_vector_start
        vector_time_sum += vector_time
        
        vector_rank = get_ground_truth_rank(ground_truth_name, vector_retrieved_names)
        if vector_rank > 0:
            vector_hits += 1
            vector_mrr_sum += 1.0 / vector_rank
        
        print(f" -> Kết quả Vector Search Top {top_k}: {vector_retrieved_names}")
        print(f"    Thành công? {'✅ CÓ (Hạng ' + str(vector_rank) + ')' if vector_rank > 0 else '❌ KHÔNG'} | Độ trễ: {vector_time*1000:.1f}ms")
        
        # Phương pháp 2: BM25 Search trên DB thực
        t_bm25_start = time.time()
        tokenized_query = search_query_str.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_retrieved_names = [names[i] for i in top_bm25_indices]
        t_bm25_end = time.time()
        
        bm25_time = t_bm25_end - t_bm25_start
        bm25_time_sum += bm25_time
        
        bm25_rank = get_ground_truth_rank(ground_truth_name, bm25_retrieved_names)
        if bm25_rank > 0:
            bm25_hits += 1
            bm25_mrr_sum += 1.0 / bm25_rank
        
        print(f" -> Kết quả BM25 Search Top {top_k}:   {bm25_retrieved_names}")
        print(f"    Thành công? {'✅ CÓ (Hạng ' + str(bm25_rank) + ')' if bm25_rank > 0 else '❌ KHÔNG'} | Độ trễ: {bm25_time*1000:.1f}ms")
        
        time.sleep(1) # Tránh rate limit của API
        
    print("\n" + "="*80)
    print("BÁO CÁO KẾT QUẢ SO SÁNH CHẤT LƯỢNG (K=5) TRÊN DATABASE THỰC TẾ")
    print("="*80)
    print(f"Tổng số câu hỏi đánh giá hợp lệ: {len(test_indices)}")
    print(f"\n1. VECTOR SEARCH (Semantic Search - LL_Keywords vs Vector[Name+Description])")
    print(f"   - Hit Rate @ 5 : {vector_hits}/{len(test_indices)} ({(vector_hits/len(test_indices))*100:.1f}%)")
    print(f"   - MRR (Chất lượng xếp hạng)  : {vector_mrr_sum / len(test_indices):.4f}")
    print(f"   - Avg Latency  : {(vector_time_sum / len(test_indices))*1000:.1f} ms")
    
    print(f"\n2. BM25 SEARCH (Lexical Search - LL_Keywords vs Text[Name])")
    print(f"   - Hit Rate @ 5 : {bm25_hits}/{len(test_indices)} ({(bm25_hits/len(test_indices))*100:.1f}%)")
    print(f"   - MRR (Chất lượng xếp hạng)  : {bm25_mrr_sum / len(test_indices):.4f}")
    print(f"   - Avg Latency  : {(bm25_time_sum / len(test_indices))*1000:.1f} ms")
    print("="*80)

if __name__ == "__main__":
    evaluate_search_methods(sample_size=10)
