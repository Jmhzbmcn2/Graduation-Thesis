import os
import json
import requests
import sys

# Thêm đường dẫn để import được từ eval script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_ragas_hybrid_mix_beam import extract_chunks_from_context

LIGHTRAG_URL = "http://localhost:9621"

def group_linearize_context(raw_context: str) -> str:
    import json, re
    
    # Gom nhóm Graph Data
    grouped_data = {}
    
    entity_start = raw_context.find("Knowledge Graph Data (Entity):")
    rel_start = raw_context.find("Knowledge Graph Data (Relationship):")
    chunk_start = raw_context.find("Document Chunks")
    
    if entity_start != -1:
        end_idx = rel_start if rel_start != -1 else chunk_start
        entity_section = raw_context[entity_start:end_idx]
        json_match = re.search(r'```json\s*\n(.*?)```', entity_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        name = obj.get("entity", "")
                        desc = obj.get("description", "")
                        if name:
                            grouped_data[name] = {"description": desc, "relations": []}
                    except Exception:
                        pass
                        
    if rel_start != -1:
        end_idx = chunk_start if chunk_start != -1 else len(raw_context)
        rel_section = raw_context[rel_start:end_idx]
        json_match = re.search(r'```json\s*\n(.*?)```', rel_section, re.DOTALL)
        if json_match:
            for line in json_match.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        e1 = obj.get("entity1", "")
                        e2 = obj.get("entity2", "")
                        desc = obj.get("description", "")
                        if e1 and e2 and desc:
                            if e1 not in grouped_data:
                                grouped_data[e1] = {"description": "", "relations": []}
                            if e2 not in grouped_data:
                                grouped_data[e2] = {"description": "", "relations": []}
                            
                            grouped_data[e1]["relations"].append(f"Quan hệ với [{e2}]: {desc}")
                            grouped_data[e2]["relations"].append(f"Quan hệ với [{e1}]: {desc}")
                    except Exception:
                        pass
                        
    # Build phần thực thể
    graph_lines = ["DỮ LIỆU ĐỒ THỊ TRI THỨC (GRAPH CONTEXT):"]
    if grouped_data:
        for i, (name, data) in enumerate(grouped_data.items(), 1):
            graph_lines.append(f"\n{i}. Thực thể: [{name}]")
            if data["description"]:
                graph_lines.append(f"   - Thông tin: {data['description']}")
            for rel in data["relations"]:
                graph_lines.append(f"   - {rel}")
                
    entities_str = "\n".join(graph_lines)
    
    # Trích xuất nguyên vẹn phần Document Chunks và Reference Document List
    rest_of_context = ""
    if chunk_start != -1:
        rest_of_context = "\n\n" + raw_context[chunk_start:].strip()
        
    return entities_str + rest_of_context

def query_and_save_context(question: str, mode: str, output_file: str):
    print(f"\n[{mode.upper()}] Querying LightRAG...")
    payload = {
        "query": question,
        "mode": mode,
        "only_need_context": True  # Cờ này giúp LightRAG trả về luôn context thay vì sinh ra câu trả lời
    }
    
    try:
        response = requests.post(f"{LIGHTRAG_URL}/query", json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            context = data.get("response", "NO CONTEXT RETURNED")
            
            # Tránh lỗi None khi key context tồn tại nhưng value là null
            if not context:
                context = str(data)
            
            # Mô phỏng quá trình xử lý (Tuyến tính hóa) trước khi feed vào LLM
            if mode == "beam":
                # Đối với Beam: Dữ liệu đưa vào LLM của LightRAG là dữ liệu ĐÃ ĐƯỢC PARSE
                context = group_linearize_context(context)
                ragas_context = context
            else:
                # Đối với Hybrid/Mix: Dữ liệu đưa vào LLM vẫn là RAW JSON
                ragas_context = extract_chunks_from_context(context)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"QUESTION: {question}\n")
                f.write(f"MODE: {mode.upper()}\n")
                f.write("="*50 + "\n")
                f.write("CONTEXT ĐƯA VÀO LLM CỦA LIGHTRAG (Sau khi tiền xử lý):\n")
                f.write("="*50 + "\n")
                f.write(context)
                f.write("\n\n")
                f.write("="*50 + "\n")
                f.write("CONTEXT CHO RAGAS (LLM Judge):\n")
                f.write("="*50 + "\n")
                f.write(ragas_context)
                f.write("\n")
                
            print(f"OK: Da luu raw context cua {mode} vao {output_file}")
            return context
        else:
            print(f"Loi HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Loi ket noi: {e}")

if __name__ == "__main__":
    question = "Có bắt buộc phải sử dụng thuốc Carbogast trong khi mang thai và cho con bú không? Ghi chú các điều kiện cụ thể."
    
    os.makedirs("testcase/contexts", exist_ok=True)
    
    query_and_save_context(question, "hybrid", "testcase/contexts/context_hybrid.txt")
    query_and_save_context(question, "mix", "testcase/contexts/context_mix.txt")
    query_and_save_context(question, "beam", "testcase/contexts/context_beam.txt")
    
    print("\nHoan thanh! Hay mo thu muc testcase/contexts/ de xem truc tiep cac file txt.")
