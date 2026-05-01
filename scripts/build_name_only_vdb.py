"""
Phase 1: Build Name-Only Entity VectorDB (Offline - One Time)

Script này đọc toàn bộ entity names từ vdb_entities.json hiện tại,
gọi Ollama để tính embedding chỉ cho tên (name) của mỗi entity,
và lưu kết quả vào file vdb_entities_name_only.json riêng biệt.

File này KHÔNG chạm vào database gốc.

Sử dụng: python scripts/build_name_only_vdb.py
"""

import asyncio
import json
import os
import sys
import time
import base64
import zlib

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

WORKING_DIR = "medical_rag_v6"
VDB_ENTITIES_PATH = os.path.join(WORKING_DIR, "vdb_entities.json")
OUTPUT_PATH = os.path.join(WORKING_DIR, "vdb_entities_name_only.json")

# Embedding config — uses the same Ollama model as LightRAG
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
BATCH_SIZE = 64  # Số lượng tên gửi cho Ollama mỗi lần


async def get_ollama_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Gọi Ollama API để tính embedding cho 1 batch text."""
    import httpx

    embeddings = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for text in texts:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
    return np.array(embeddings, dtype=np.float32)


async def build_name_only_vdb():
    """Main build logic."""
    print(f"[1/3] Doc entity names tu: {VDB_ENTITIES_PATH}")
    t0 = time.time()

    with open(VDB_ENTITIES_PATH, "r", encoding="utf-8") as f:
        vdb_data = json.load(f)

    items = vdb_data.get("data", []) or vdb_data.get("__data__", [])
    if isinstance(items, dict):
        items = list(items.values())

    # Trích xuất entity_name và __id__
    entity_names = []
    entity_ids = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("entity_name", "")
        eid = item.get("__id__", "")
        if name and eid:
            entity_names.append(name)
            entity_ids.append(eid)

    total = len(entity_names)
    print(f"[OK] Tim thay {total} entities. Bat dau tinh embedding cho ten...")

    # Tính embedding theo batch
    output_records = []
    for i in range(0, total, BATCH_SIZE):
        batch_names = entity_names[i : i + BATCH_SIZE]
        batch_ids = entity_ids[i : i + BATCH_SIZE]

        embeddings = await get_ollama_embeddings_batch(batch_names)

        for j, (name, eid, emb) in enumerate(
            zip(batch_names, batch_ids, embeddings)
        ):
            # Nén vector theo cùng format với LightRAG gốc (float16 + zlib + base64)
            vector_f16 = emb.astype(np.float16)
            compressed = zlib.compress(vector_f16.tobytes())
            encoded = base64.b64encode(compressed).decode("utf-8")

            output_records.append(
                {
                    "__id__": eid,
                    "entity_name": name,
                    "content": name,  # Content chỉ là tên, không gộp description
                    "vector": encoded,
                    "__vector__": emb.tolist(),
                    "__created_at__": int(time.time()),
                }
            )

        elapsed = time.time() - t0
        done = min(i + BATCH_SIZE, total)
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        print(
            f"  [{done}/{total}] ({done/total*100:.1f}%) | "
            f"Speed: {speed:.1f} entities/s | "
            f"ETA: {eta:.0f}s"
        )

    # Lưu ra file JSON (cùng format với NanoVectorDB)
    output_data = {"data": output_records}

    print(f"\n[2/3] Dang luu vao: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"[DONE] HOAN TAT BUILD NAME-ONLY VECTOR DB!")
    print(f"   Total entities : {total}")
    print(f"   File size      : {file_size_mb:.1f} MB")
    print(f"   Time elapsed   : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Output         : {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(build_name_only_vdb())
