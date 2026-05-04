import os
import re
import time
import warnings
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)

warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*token usage.*", category=UserWarning)

# ======================== CẤU HÌNH ========================
LIGHTRAG_URL = "http://localhost:9621"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_SCRIPT_DIR, "300_case_random.csv")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "eval_focused_chunk_only.xlsx")

EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ"))
EVAL_LLM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "EMPTY")
EVAL_LLM_HOST = os.getenv("EVAL_LLM_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "http://localhost:8000/v1"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
EMBEDDING_HOST  = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

TEST_LIMIT = 30
MAX_CONTEXT_CHARS = None

MODES = ["focused"]
EVAL_BATCH_SIZE = 100

# ======================== CẤU HÌNH FOCUSED ========================
FOCUSED_TOP_K = 15                          # Số anchor nodes
FOCUSED_EDGE_QUOTA = 10                     # Max edges per anchor
FOCUSED_EDGE_THRESHOLD = 0.3               # Min semantic score
FOCUSED_ALPHA = 0.3                        # Weight anchor score
FOCUSED_BETA = 0.7                         # Weight edge semantic score
FOCUSED_MAX_EDGES = 50                     # Global cap edges
FOCUSED_CHUNK_TOP_K = 15                   # chunk_top_k (tăng lên 15 do tiết kiệm được token từ Entity/Relation)
FOCUSED_ANCHOR_TOP_K = 10                   # K per-branch per-keyword (BM25 & Semantic)
FOCUSED_ANCHOR_SEMANTIC_THRESHOLD = 0.6    # Ngưỡng cosine Branch 2 (Semantic)
FOCUSED_BOTH_BONUS = 0.1                   # Bonus for entities found by BOTH BM25 + semantic
FOCUSED_CHUNK_TOP_K_RERANK = 10             # Top-K chunks sau rerank (tăng lên 15)

# ======================== TOGGLE CONTEXT ========================
# True  → truyền entities/relations vào context LLM
# False → để trống (chỉ dùng chunks, như hiện tại)
INCLUDE_ENTITIES  = False
INCLUDE_RELATIONS = False
# ================================================================

# ===========================================================

def load_existing_results() -> dict:
    existing = {}
    if not os.path.exists(OUTPUT_FILE):
        return existing
    try:
        xls = pd.ExcelFile(OUTPUT_FILE, engine="openpyxl")
        for mode in MODES:
            sheet = mode.capitalize()[:31]
            if sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                if "question_text" in df.columns and len(df) > 0:
                    existing[mode] = df
                    print(f"   ♻️  Đã tải {len(df)} kết quả cũ cho mode {mode.upper()}")
        xls.close()
    except Exception as e:
        print(f"   ⚠️ Không đọc được file kết quả cũ: {e}")
    return existing

def clean_answer(answer: str) -> str:
    return re.split(r"\n*###\s*References", answer, maxsplit=1)[0].strip()

def estimate_tokens(text: str) -> int:
    if not text: return 0
    return max(1, len(text) // 4)

def truncate_context(text: str, max_chars: int = None) -> str:
    if max_chars is None: return text
    if len(text) <= max_chars: return text
    return text[:max_chars] + "\n... [truncated]"

def query_lightrag_with_timing(question: str, mode: str, retries: int = 3):
    from lightrag.prompt import PROMPTS
    base = {
        "query": question,
        "mode": mode,
        "top_k": FOCUSED_TOP_K,
        "chunk_top_k": FOCUSED_CHUNK_TOP_K,
        "focused_edge_quota": FOCUSED_EDGE_QUOTA,
        "focused_edge_threshold": FOCUSED_EDGE_THRESHOLD,
        "focused_alpha": FOCUSED_ALPHA,
        "focused_beta": FOCUSED_BETA,
        "focused_max_edges": FOCUSED_MAX_EDGES,
        "focused_anchor_top_k": FOCUSED_ANCHOR_TOP_K,
        "focused_anchor_semantic_threshold": FOCUSED_ANCHOR_SEMANTIC_THRESHOLD,
        "focused_both_bonus": FOCUSED_BOTH_BONUS,
        "focused_chunk_top_k": FOCUSED_CHUNK_TOP_K_RERANK,
        "enable_rerank": True,
    }

    for attempt in range(retries):
        try:
            # 1. Gọi API /query/data để CHỈ LẤY DATA (không gọi LLM ở server)
            client_start = time.perf_counter()
            resp = requests.post(f"{LIGHTRAG_URL}/query/data", json=base, timeout=180)
            
            resp.raise_for_status()
            resp_json = resp.json()
            
            # 2. Xây dựng Context sử dụng ĐÚNG template kg_query_context của gốc
            import json as _json
            chunks_data = resp_json.get("data", {}).get("chunks", [])
            text_chunks = []
            for c in chunks_data:
                text_chunks.append({"reference_id": c.get("reference_id", ""), "content": c.get("content", "")})
            text_chunks_str = "\n".join(_json.dumps(c, ensure_ascii=False) for c in text_chunks)

            # Build entities_str — map entity_name→entity, entity_type→type
            if INCLUDE_ENTITIES:
                entities_data = resp_json.get("data", {}).get("entities", [])
                entities_context = [
                    {"entity": e.get("entity_name", ""), "type": e.get("entity_type", ""), "description": e.get("description", "")}
                    for e in entities_data
                ]
                entities_str = "\n".join(_json.dumps(e, ensure_ascii=False) for e in entities_context)
            else:
                entities_str = ""

            # Build relations_str — map src_id→entity1, tgt_id→entity2
            if INCLUDE_RELATIONS:
                relations_data = resp_json.get("data", {}).get("relationships", [])
                relations_context = [
                    {"entity1": r.get("src_id", ""), "entity2": r.get("tgt_id", ""), "description": r.get("description", "")}
                    for r in relations_data
                ]
                relations_str = "\n".join(_json.dumps(r, ensure_ascii=False) for r in relations_context)
            else:
                relations_str = ""
            
            references_data = resp_json.get("data", {}).get("references", [])
            reference_list_str = "\n".join(f"[{r.get('reference_id', '')}] {r.get('file_path', '')}" for r in references_data)
            
            # Đổ dữ liệu vào đúng template gốc, nhưng BỎ TRỐNG entities và relations
            kg_context_template = PROMPTS["kg_query_context"]
            full_context = kg_context_template.format(
                entities_str=entities_str,
                relations_str=relations_str,
                text_chunks_str=text_chunks_str,
                reference_list_str=reference_list_str
            )
                
            full_context = truncate_context(full_context, MAX_CONTEXT_CHARS)
            contexts = [full_context]
            
            metadata = resp_json.get("metadata", {})
            retrieval_latency_ms = (time.perf_counter() - client_start) * 1000
            
            # 3. Format System Prompt y hệt LightRAG core
            sys_prompt_temp = PROMPTS.get("rag_response", "You are a helpful assistant answering questions based on the provided context.")
            sys_prompt = sys_prompt_temp.format(
                response_type="Multiple Paragraphs",
                user_prompt="n/a",
                context_data=full_context
            )
            
            # Câu hỏi chỉ là text thuần túy (Context đã được nhúng vào System Prompt)
            user_prompt_llm = question
            
            llm_payload = {
                "model": EVAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt_llm}
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
                "stream": False
            }
            
            llm_start = time.perf_counter()
            llm_resp = requests.post(f"{EVAL_LLM_HOST}/chat/completions", json=llm_payload, headers={"Authorization": f"Bearer {EVAL_LLM_API_KEY}"}, timeout=300)
            llm_resp.raise_for_status()
            llm_json = llm_resp.json()
            
            generation_latency_ms = (time.perf_counter() - llm_start) * 1000
            client_elapsed_ms = retrieval_latency_ms + generation_latency_ms
            
            answer = llm_json["choices"][0]["message"]["content"]
            answer = clean_answer(answer)
            
            # Tính toán metrics
            input_tokens = llm_json.get("usage", {}).get("prompt_tokens") or estimate_tokens(user_prompt + sys_prompt)
            output_tokens = llm_json.get("usage", {}).get("completion_tokens") or estimate_tokens(answer)

            metrics = {
                "keyword_extraction_ms": 0, # /query/data không trả về detail timing
                "graph_search_ms": 0,
                "rerank_ms": 0,
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "rerank_latency_ms": 0,
                "generation_latency_ms": round(generation_latency_ms, 2),
                "total_latency_ms": round(client_elapsed_ms, 2),
                "client_wall_ms": round(client_elapsed_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

            return answer, contexts, metrics

        except Exception as e:
            print(f"    ⚠️ Lỗi (lần {attempt+1}/{retries}): {e}")
            if attempt < retries - 1: time.sleep(3)

    return "Error: Không lấy được response", ["No context"], {k: 0 for k in ["keyword_extraction_ms", "graph_search_ms", "rerank_ms", "retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms", "input_tokens", "output_tokens", "total_latency_ms", "client_wall_ms"]}

def run_ragas_evaluation(questions, answers, contexts_list, ground_truths):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_HOST,
            temperature=0.0,
            max_tokens=4096,
            max_retries=3,
            timeout=300.0,
        ),
        bypass_n=True,
    )

    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key="ollama",
            base_url=f"{EMBEDDING_HOST}/v1",
            check_embedding_ctx_length=False,
            max_retries=5,
            timeout=60.0,
        )
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    from ragas.run_config import RunConfig
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=4, max_retries=10),
    )
    return results.to_pandas()

def main():
    print("=" * 70)
    print("📊 RAGAS Evaluation — Mode FOCUSED (Chunk-Only Context)")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT)
    
    all_questions, all_ground_truths = [], []
    for _, row in df.iterrows():
        question = str(row.get("question", row.get("Question", "")))
        ground_truth = str(row.get("answer", row.get("Answer", "")))
        if question and question != "nan":
            all_questions.append(question)
            all_ground_truths.append(ground_truth)

    total = len(all_questions)
    existing_results = load_existing_results()
    all_mode_results = {}

    for mode in MODES:
        existing_df = existing_results.get(mode)
        done_questions = set(existing_df["question_text"].tolist()) if existing_df is not None else set()
        new_indices = [i for i, q in enumerate(all_questions) if q not in done_questions]

        if not new_indices:
            print("✅ Tất cả câu đã có kết quả!")
            if existing_df is not None:
                print("\n📊 BẢNG KẾT QUẢ ĐÁNH GIÁ (TRUNG BÌNH):")
                eval_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
                for col in eval_cols:
                    if col in existing_df.columns:
                        print(f"   - {col:18s}: {existing_df[col].mean():.4f}")
                print()
            continue

        print(f"\n🔄 Bắt đầu chạy truy vấn cho {len(new_indices)} câu mới...")
        mode_questions, mode_answers, mode_contexts_list, mode_ground_truths, mode_metrics = [], [], [], [], []

        for idx, i in enumerate(new_indices):
            print(f"\n[{idx+1}/{len(new_indices)}] Câu hỏi: {all_questions[i][:60]}...")
            ans, ctx, met = query_lightrag_with_timing(all_questions[i], mode)
            print(f"   Answer: {ans[:80]}...")
            mode_questions.append(all_questions[i])
            mode_answers.append(ans)
            mode_contexts_list.append(ctx)
            mode_ground_truths.append(all_ground_truths[i])
            mode_metrics.append(met)

        timing_df = pd.DataFrame(mode_metrics)
        timing_df.insert(0, "question_text", mode_questions)

        print("\n🔬 Chạy RAGAS evaluation...")
        try:
            ragas_df = run_ragas_evaluation(mode_questions, mode_answers, mode_contexts_list, mode_ground_truths)
            ragas_df.insert(0, "question_text", mode_questions)
            
            valid_ragas = [c for c in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"] if c in ragas_df.columns]
            ragas_df["ragas_score"] = ragas_df[valid_ragas].mean(axis=1)

            for col in timing_df.columns:
                if col != "question_text":
                    ragas_df[col] = timing_df[col].values

            new_results_df = ragas_df
        except Exception as e:
            print(f"❌ Lỗi RAGAS: {e}")
            new_results_df = timing_df

        if existing_df is not None:
            final_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        else:
            final_df = new_results_df

        all_mode_results[mode] = final_df

        print("\n📊 BẢNG KẾT QUẢ ĐÁNH GIÁ (TRUNG BÌNH):")
        eval_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "ragas_score"]
        for col in eval_cols:
            if col in final_df.columns:
                print(f"   - {col:18s}: {final_df[col].mean():.4f}")
        print()

        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name=mode.capitalize()[:31], index=False)
        print(f"💾 Đã lưu kết quả vào: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
